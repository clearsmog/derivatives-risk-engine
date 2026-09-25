# MAR21 parameters used by `frtb.py`

Transcribed from the BCBS Basel Framework, MAR21 (version in force from
1 Jan 2023, published 27 Mar 2020), bis.org, read 2026-09-25. Only the pieces
the engine uses are listed. Where the text leaves a choice to the bank, the
choice made here is stated.

## General (MAR21.4-7)

- Kb = sqrt(max(0, sum WS_k^2 + sum_{k != l} rho_kl WS_k WS_l)).
- Across buckets: sqrt(sum Kb^2 + sum_{b != c} gamma_bc S_b S_c), S_b = sum WS_k;
  if the sum under the root is negative, S_b = max(min(sum WS_k, Kb), -Kb).
- Correlation scenarios: medium as given; high = min(1.25 rho, 100%);
  low = max(2 rho - 100%, 75% rho). Capital is the maximum of the three totals.
- Curvature (MAR21.5): CVR_k^± = -sum_i [V_i(x^{±RW}) - V_i(x) ∓ RW * s_ik];
  Kb = max(Kb^+, Kb^-) with psi = 0 when both CVRs are negative; correlations
  are the squared delta ones (MAR21.100-101). Here the scenario adjustment is
  applied to the squared value.

## Sensitivity definitions (MAR21.19-25)

| Risk class | Delta | Units of the shift |
|---|---|---|
| GIRR | PV01: (V(r + 1bp) - V(r)) / 0.0001 per tenor | absolute |
| CSR | CS01: (V(cs + 1bp) - V(cs)) / 0.0001 per tenor | absolute |
| Equity, commodity, FX | (V(1.01 x) - V(x)) / 0.01 | relative |
| Vega (all) | vega * implied vol | |

GIRR and CSR vega may use normal or lognormal vol (MAR21.28); equity,
commodity and FX must use lognormal. Swaptions here use normal vol.

## GIRR (MAR21.8, 21.41-50)

| Tenor (y) | 0.25 | 0.5 | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| RW | 1.7% | 1.7% | 1.6% | 1.3% | 1.2% | 1.1% | 1.1% | 1.1% | 1.1% | 1.1% |

- Bucket = currency. Tenor correlation Table 2 (same curve), e.g.
  0.25y-10y 40.0%, 2y-5y 95.6%, 5y-10y 97.0%. Different curves, same tenor:
  99.9%; different curves and tenors: Table 2 x 99.9%.
- Across currencies: gamma = 50%.
- The optional sqrt(2) reduction for specified currencies (MAR21.44) is not
  applied.
- Vega factors: option maturity and underlying residual maturity, each on
  {0.5, 1, 3, 5, 10}; rho = min(rho_opt * rho_und, 1), both
  exp(-1% |Ti - Tj| / min(Ti, Tj)) (MAR21.93).
- Curvature: parallel shift of the whole curve by the highest tenor RW, 1.7%
  (MAR21.99).

## CSR non-securitisation (MAR21.9, 21.51-57)

| Bucket | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RW | 0.5% | 1.0% | 5.0% | 3.0% | 3.0% | 2.0% | 1.5% | 2.5% | 2.0% | 4.0% | 12.0% | 7.0% | 8.5% | 5.5% | 5.0% | 12.0% | 1.5% | 5.0% |

- Buckets 1-8 IG, 9-15 HY/NR by sector; 4/12 basic materials, energy,
  industrials; 6/14 technology, telecoms; 16 other; 17 IG indices; 18 HY indices.
- Tenors {0.5, 1, 3, 5, 10}, same RW for all tenors.
- Within bucket (1-15): rho = rho_name (35%) x rho_tenor (65%) x rho_basis
  (99.9% bond vs CDS curve). Buckets 17-18: rho_name 80%.
- Across buckets: gamma = gamma_rating x gamma_sector. MAR21.57: gamma_rating
  is 50% "where the two buckets b and c are both in buckets 1 to 15 and have a
  different rating category (either IG or HY/NR)", and 1 otherwise, so an HY
  single name against an IG index (14 vs 17) takes gamma_rating = 1.
  gamma_sector from Table 5, e.g. 4/12 vs 6/14: 25%; 4/12 vs 7/15: 5%; any of
  1-15 vs 17 or 18: 45%; 17 vs 18: 75%.
- Bucket 16 (simple sum of absolute values) is not used here.

## Equity (MAR21.12, 21.72-80)

| Bucket | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Spot RW | 55% | 60% | 45% | 55% | 30% | 35% | 40% | 50% | 70% | 50% | 70% | 15% | 25% |

- Bucket 8: large cap, advanced economy, financials / real estate / technology.
  Bucket 12: large-cap advanced-economy indices; 13: other indices.
- Within bucket, spot vs spot: 15% (buckets 1-4), 25% (5-8), 7.5% (9),
  12.5% (10), 80% (12-13).
- Across buckets: 15% within 1-10; 0% with 11; 75% between 12 and 13; 45%
  otherwise.
- Repo-rate risk factors are not modelled.
- Vega RW: large cap and indices LH 20, so min(55% sqrt(2), 100%) = 77.78%;
  small cap and other LH 60, 100%.
- Curvature RW = delta RW, relative spot shift (MAR21.98).

## Commodity (MAR21.13, 21.81-85)

Bucket RW: 1 30%, 2 35%, 3 60%, 4 80%, 5 40%, 6 45%, 7 20%, 8 35%, 9 25%,
10 35%, 11 50%. rho_cty: 55, 95, 40, 80, 60, 65, 55, 45, 15, 40, 15%.
rho_tenor 99%, rho_basis 99.9%, gamma 20% (0% with bucket 11). Tenors
{0, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}. Vega RW 100% (LH 120). Curvature:
parallel shift of the whole curve by the bucket RW.

## FX (MAR21.14, 21.86-89)

- Bucket = each currency against the reporting currency (USD here).
- RW 15%; the optional sqrt(2) reduction for specified pairs (MAR21.88) is not
  applied. Across buckets: gamma = 60%.
- Vega RW: LH 40, 100%. Curvature RW 15%; the optional 1.5 scalar is not applied.

## MAR32.42 (PLA test)

Green: Spearman > 0.80 and KS < 0.09. Red: Spearman < 0.70 or KS > 0.12.
Amber otherwise.
