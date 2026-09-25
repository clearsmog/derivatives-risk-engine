# derivatives-risk-engine

A daily market-risk batch for a multi-asset derivatives book: pricing, Greeks,
historical VaR and ES by full revaluation, stress tests, P&L explain, the FRTB
P&L attribution test, and FRTB standardised-approach capital across GIRR,
credit spread, equity, FX and commodity risk. Results load into two
[Atoti](https://www.atoti.io) cubes: one for VaR and stress by desk, book,
counterparty or asset class, one for sensitivities by risk factor and tenor.

The book and the market history are synthetic. Every name, size and price is
invented.

```
uv sync
uv run pytest                              # 59 tests; the cube test skips without Atoti
uv run python scripts/run_demo.py          # risk pack -> outputs/risk_pack.md
uv sync --extra cube
uv run --extra cube python -m derivatives_risk_engine.cube   # cubes + web UI
```

## The book

| Desk | Positions |
|---|---|
| COMMODITIES | Brent-indexed LNG supply contracts resold at JKM / TTF; a zero-cost collar, capped swap and fixed swap sold to downstream buyers; partial bank hedges (swaps, Asian calls, Brent options) |
| EQUITY | SPX 3m puts paid for by selling 1y calls (zero premium); a Euro Stoxx put (EUR, so it carries FX); an NWT down-and-out call; forwards sized to hedge SPX and NWT delta |
| RATES | A PV01-neutral USD 2s10s steepener, a 1y x 10y payer swaption, a EUR 5y receiver |
| FX | A long EUR forward, a short 3m EUR call, a long 1y EUR put |
| CREDIT | ACME 5y protection bought against CDX IG protection sold, CS01-neutral; ZETA 3y (high yield) protection bought |

## What it computes

| Module | Does |
|---|---|
| `market.py` | Every input is a named risk factor (price, rate, spread or vol) with pillars, so bumps, scenarios and FRTB mapping work the same way for every asset class |
| `pricing.py` | Black-76; arithmetic Asians by lognormal moment matching (checked against Monte Carlo); a down-and-out call by the reflection formula (checked against Monte Carlo with the Broadie-Glasserman-Kou shift); Bachelier swaptions; CDS on a hazard curve bootstrapped to reprice every quote |
| `instruments/` | Commodity swaps, Asians and futures options; equity options, forwards and a barrier; FX forwards and Garman-Kohlhagen options; OIS swaps and swaptions; CDS. Each leg declares the risk-factor pillars it depends on |
| `sensitivities.py` | Bump-and-reprice first and second derivatives per leg and pillar: cash delta and gamma, PV01, CS01, vega, volga, theta; plus the MAR21 FRTB sensitivities |
| `scenarios.py` | A 500-day correlated multi-asset history (Student-t days, GARCH clustering, curve factors, a 60-day stress window) and six named stresses |
| `var.py` | Full-revaluation VaR 99% and ES 97.5%, and Greeks-based proxies to compare against |
| `pnl_explain.py` | HPL against RTPL split into delta, gamma, vega, theta and unexplained; the MAR32 PLA test per desk against three risk models |
| `frtb.py` | MAR21 sensitivities-based method for GIRR, CSR non-securitisation, equity, FX and commodity: delta, vega and curvature, three correlation scenarios, per desk and firm-wide. Parameters with paragraph references in [`docs/mar21-parameters.md`](docs/mar21-parameters.md) |
| `cube.py` | Two Atoti cubes. VaR and ES are taken on summed P&L vectors, so they aggregate correctly at any level and match Python to the cent |

## Results on the sample book (USD)

**VaR and diversification.**

| Desk | VaR 99% 1d | ES 97.5% 1d |
|---|--:|--:|
| COMMODITIES | 53.1m | 57.8m |
| CREDIT | 12.8m | 11.5m |
| FX | 11.1m | 11.2m |
| RATES | 9.0m | 10.8m |
| EQUITY | 5.3m | 5.7m |
| **Firm** | **55.0m** | **57.0m** |
| Sum of desks | 91.3m | 96.9m |

**Greeks against full revaluation.** At firm level the book is dominated by
linear commodity and FX delta, and a Greeks proxy lands within 1.5% of full
revaluation (delta only 55.8m, adding gamma 54.8m, adding vega 55.2m, full
55.0m). The equity desk is different: once its delta is hedged, vega explains
more of its daily P&L than delta does (standard deviation 1.8m against 1.0m
over 250 days), and cross terms such as spot-vol leave 0.13m a day
unexplained.

**Stress.** The firm gains in a 2022-style gas squeeze (+955m, from the
supply contracts' long-gas, short-oil shape) and loses in an oil spike with
gas flat (-182m). In a 2020-style crash the equity desk's puts and
long-vega position gain 195m and the credit desk's net protection gains 112m.

**P&L attribution (Spearman / KS, 250 out-of-sample days).**

| Desk | Full risk factors | One factor per curve, fitted betas | Anchor copied onto the curve |
|---|---|---|---|
| COMMODITIES | 1.000 / 0.012 green | 0.936 / 0.056 green | 0.927 / 0.160 red |
| CREDIT | 1.000 / 0.012 green | 1.000 / 0.012 green | 0.999 / 0.024 green |
| EQUITY | 0.999 / 0.020 green | 0.982 / 0.036 green | 0.976 / 0.076 green |
| FX | 1.000 / 0.012 green | 0.977 / 0.028 green | 0.956 / 0.060 green |
| RATES | 1.000 / 0.012 green | 0.422 / 0.196 red | 0.432 / 0.192 red |

- The full model uses the same pricer and risk factors as HPL, so it only
  measures Taylor truncation and passes trivially.
- The naive proxy fails the commodity desk on KS alone: it ranks days almost
  as well as the fitted model but copies front-month volatility onto back
  months that move less, so its P&L distribution is too wide.
- The rates desk fails both proxies because it is a PV01-neutral steepener: a
  one-factor model sees the level of the curve and the desk has no level
  risk. In the synthetic history the slope factor pivots at 5y, which is also
  the proxy's anchor pillar, so this failure is built in; it shows the
  mechanism rather than a finding about real markets.

**FRTB SA capital.** 1,066m firm-wide, binding in the low correlation
scenario.

| Risk class | Delta | Vega | Curvature | Total |
|---|--:|--:|--:|--:|
| Commodity | 396.6m | 12.3m | 0 | 408.9m |
| CSR non-sec | 321.8m | 0 | 0 | 321.8m |
| GIRR | 133.6m | 31.8m | 0 | 165.5m |
| FX | 95.2m | 10.8m | 25.8m | 131.9m |
| Equity | 9.9m | 28.0m | 0 | 37.9m |

- **Credit basis gets little relief.** The ACME vs CDX position is
  CS01-neutral, but the single name (bucket 4) and the index (bucket 17) only
  offset at the 45% cross-bucket correlation, so the charge is close to the
  gross position.
- **FX curvature** (25.8m) comes from the short ATM EUR call.
- **TTF and JKM** are treated as one commodity at two delivery locations by
  default; as distinct commodities the firm figure is 1,038m.
- **Scale.** Risk weights are calibrated to stressed ES over liquidity
  horizons of 10 to 120 days, not to a daily move, so SA capital sitting well
  above 1-day ES is expected.

## Conventions and choices worth knowing

- **Sensitivities.** For each leg and pillar the engine stores d1 and d2 in
  the units the scenarios move: simple return for prices (cash delta, cash
  gamma), absolute change for rates, spreads and vols. The Taylor P&L is then
  one formula for every asset class. FRTB sensitivities follow MAR21.19-25:
  one-sided 1bp for PV01 and CS01, one-sided 1% for equity, FX and commodity,
  vega times implied vol (normal vol for swaptions, which MAR21.28 allows).
- **Rates** are single-curve OIS: the zero curve both discounts and projects,
  floating legs compound the overnight rate, fixed legs pay annually on plain
  year fractions.
- **Credit.** Hazard rates are piecewise constant and bootstrapped so each
  pillar CDS prices at its quote; premium accrued to default is included.
  Simplified against the ISDA standard model (no day-count or step-in
  conventions).
- **Vol** is an ATM term structure with no smile, interpolated in total
  variance and flat beyond the last pillar; the barrier option is therefore
  priced without skew.
- **FRTB choices** the text leaves open are listed at the top of `frtb.py`.
  One reading worth knowing: MAR21.57 sets gamma_rating to 50% only between
  buckets 1 to 15 of different rating, so an HY name against the IG index
  bucket takes the full 45% sector gamma.
- **VaR quantile.** The loss at rank `floor((n - 1) * 1%)` in ascending order
  (numpy `lower`, Atoti `mode="inc", interpolation="lower"`). ES averages the
  worst `floor(n * 2.5%)` days, so ES 97.5% can come out below VaR 99% for a
  desk with a short, sharp tail.
- **P&L explain does not age the book** along the test path: each day's move
  is applied to today's positions.

## Limitations

- Synthetic market data; `synthetic_history` returns the same shape a real
  history would.
- No smile or skew anywhere; the down-and-out call would be worth less under
  equity skew.
- FX forwards use covered interest parity with no cross-currency basis.
- Oil indexation uses same-month Brent (real contracts lag, e.g. 3-0-1), and
  commodity Asians average one contract's forward over its own delivery month.
- Out of scope: CSR securitisation, equity repo, inflation and cross-currency
  basis risk factors, the default risk charge, the residual risk add-on (the
  barrier and Asian options would likely attract it), the internal models
  approach beyond PLA, and CVA.
- The Atoti community edition is free for non-commercial use, and each release
  stops working 100 days after it comes out. Upgrade the package if the cube
  refuses to start.

## References

- BCBS Basel Framework: MAR21 (sensitivities-based method, version in force
  from 1 Jan 2023) and MAR32 (backtesting and P&L attribution).
- Black, F. (1976). The pricing of commodity contracts. *JFE* 3.
- Levy, E. (1992). Pricing European average rate currency options. *JIMF* 11(5).
- Reiner, E. and Rubinstein, M. (1991). Breaking down the barriers. *Risk* 4(8).
- Broadie, M., Glasserman, P. and Kou, S. (1997). A continuity correction for
  discrete barrier options. *Mathematical Finance* 7(4).
- Garman, M. and Kohlhagen, S. (1983). Foreign currency option values. *JIMF* 2(3).
- Bachelier, L. (1900). Théorie de la spéculation.
- O'Kane, D. and Turnbull, S. (2003). Valuation of credit default swaps.
  Lehman Brothers Fixed Income Quantitative Credit Research.
