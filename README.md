# commodity-risk-engine

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
uv run --extra cube python -m commodity_risk_engine.cube   # cube + web UI
```

## What it computes

| Module | Does |
|---|---|
| `pricing.py` | Black-76; arithmetic Asian options by lognormal moment matching (Levy / Turnbull-Wakeman), checked against Monte Carlo |
| `instruments.py` | Every structure is a list of three leg types (linear, Asian, European), so the risk code never needs to know what a "capped swap" is |
| `sensitivities.py` | Bump-and-reprice delta, gamma, vega and theta per leg, in both trader and FRTB conventions |
| `scenarios.py` | A 500-day synthetic history with fat tails, GARCH volatility clustering, the Samuelson effect, curve shape factors and a 60-day gas squeeze; five named stress scenarios |
| `var.py` | Full-revaluation VaR 99% and ES 97.5%, component VaR, and delta / delta-gamma / delta-gamma-vega Taylor proxies |
| `pnl_explain.py` | Hypothetical P&L against risk-theoretical P&L split into delta, gamma, vega, theta and unexplained; the MAR32 PLA test per desk |
| `frtb.py` | MAR21 sensitivities-based method, commodity risk class: delta, vega and curvature, three correlation scenarios, per desk and for the book |
| `cube.py` | Atoti cube at leg grain. Each leg carries its P&L vector, so VaR is taken on summed vectors at any level of aggregation |

## Results on the sample book (USD)

**VaR and diversification.** The supply contracts are a short-oil,
long-gas spread; the bank hedges take out most of the oil leg.

| Desk | VaR 99% 1d | ES 97.5% 1d |
|---|--:|--:|
| LNG_SUPPLY | 99.2m | 115.1m |
| BANK_HEDGES | 33.8m | 32.5m |
| CLIENT | 10.1m | 11.2m |
| **Book** | **59.8m** | **66.7m** |
| Sum of desks | 143.1m | 158.7m |

**Where a sensitivity-based view is enough, and where it isn't.** The book is
long convexity (bank options, the dealer side of the capped swap), so a
delta-only proxy overstates VaR by about 4%. Adding gamma and vega closes the
gap to full revaluation.

| Method | VaR 99% |
|---|--:|
| Delta only | 62.1m |
| Delta + gamma | 58.7m |
| Delta + gamma + vega | 59.5m |
| Full revaluation | 59.8m |

**Stress.** The book makes money in a 2022-style gas squeeze (+972m) and
loses in an oil spike with flat gas (-182m), because the supply contracts pay
an oil-linked price and earn a gas price.

**P&L attribution.** With the risk model on the full curve, every desk is
green. Replace it with a coarse model that drives each curve by its front
month, and the client and supply desks drop to amber: curve-shape and seasonal
risk show up in HPL but not in RTPL, which is the failure the PLA test is there
to catch.

| Desk | Spearman / KS (full curve) | Spearman / KS (front-month proxy) |
|---|---|---|
| BANK_HEDGES | 1.000 / 0.012 green | 0.984 / 0.088 green |
| CLIENT | 1.000 / 0.016 green | 0.999 / 0.100 amber |
| LNG_SUPPLY | 1.000 / 0.008 green | 0.992 / 0.120 amber |

**FRTB SA capital (commodity).** 382.9m for the book, binding in the high
correlation scenario, against a 1-day ES of 66.7m. The standardised risk weights
(35% for Brent, 45% for gas) sit far above any daily move. Curvature is zero for
the book, which is net long options, but 2.4m for the client desk run on its
own, which is short them. Desk-level capital sums well above the book figure
because the hedges sit on a different desk from the exposure they hedge.

## Conventions and choices worth knowing

- **Delta and gamma in log space.** `delta_ln = dV/d ln F` and
  `gamma_ln = d²V/d(ln F)²` feed the Taylor P&L on log returns. For display the
  cube shows cash gamma `F² d²V/dF² = gamma_ln - delta_ln`, which is zero for
  linear trades. The FRTB delta is the one-sided 1% bump of MAR21.23.
- **Vol is sticky-strike**, one ATM vol per contract month, no smile. MAR21.27
  allows sticky strike or sticky delta.
- **Tenor mapping.** Leg sensitivities are split linearly between the two
  adjacent MAR21.13 vertices (delta by payment date, vega by option expiry).
- **Curvature correlation scenarios.** MAR21.100-101 square the delta
  correlations; the high/low scenario adjustment is applied to the squared
  value. Some implementations adjust first and square after; the difference is
  small for this book.
- **VaR quantile.** The loss at rank `floor((n - 1) * 1%)` in ascending order
  (numpy `lower`, Atoti `mode="inc", interpolation="lower"`), so the cube and
  Python agree to the cent. ES averages the worst `floor(n * 2.5%)` days.
- **P&L explain does not age the book** along the 250-day path. Each day's
  market move is applied to today's positions, so no option fixes or expires
  mid-test and theta stays a one-day number.

## Limitations

- Synthetic market data. `synthetic_history` returns the same shape as a real
  history would; swap in a real one to use it.
- All prices in USD (TTF in USD/MMBtu), so no FX risk.
- Asian options are priced by moment matching, which carries a small bias
  against Monte Carlo. The tests bound it.
- Out of scope: FRTB default risk charge (no issuer risk in this book), the
  residual risk add-on, the internal models approach beyond the PLA test,
  credit and funding valuation adjustments.
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
