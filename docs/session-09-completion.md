# Session 09 completion

The full benchmark ran in 4.77 seconds. It reused the fixed synthetic splits and verified Session 08 propensity provenance and fold alignment. The offline fixture ran without external data. The final test splits stayed reserved.

## Synthetic validation ATE

All values are percentage points.

| Scenario | Truth | Naive | Logistic standardization | HT | Hájek | AIPW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 10.00 | 10.40 | 10.26 | 9.75 | 10.40 | 10.21 |
| C | −0.08 | 14.79 | 2.46 | −0.21 | −2.37 | −1.75 |
| F | −0.11 | 19.82 | 15.26 | 17.20 | 14.77 | 14.51 |

Training OOF AIPW estimated 0.57 pp in C against training truth −0.09 pp. F remained biased even when matching reduced the absolute engagement SMD from 1.335 to 0.006. In C, engagement and loyalty SMDs moved from 1.132 and 0.667 to −0.041 and −0.080 with ATE weights. C ATE weights had maximum 110.71 and ESS 1,246 of 4,000. Restricting C removed 418 rows and changed truth from −0.083 pp to +0.390 pp. Clipping retained all rows.

## Validation ATT

| Scenario | Full ATT truth | Regression | ATT weighting | Matched ATT | Matched truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 10.00 | 10.23 | 10.40 | −55.09 | 10.00 |
| C | 4.23 | 6.46 | 5.14 | 10.92 | 4.12 |
| F | 5.92 | 19.41 | 20.39 | 21.48 | 5.83 |

A's constant scores tied every control, and the deterministic matching rule reused one control for all 1,984 treated rows. C matched 1,722 of 1,761 treated units using 678 unique controls. Matching uncertainty is unsupported.

## Oracle nuisance simulation

Across 80 repetitions, true ATE was 10 pp. Mean AIPW estimates were 10.27 pp with both nuisances correct, 10.24 with propensity only, 10.06 with outcome only, and 25.03 with neither. Empirical coverage was 75/80, 77/80, 78/80 and 0/80 respectively. The CSV includes per-draw estimates and the summary supplies RMSE.

## NSW and CPS earnings

The publisher's RE74 variant and CPS comparison files were checksum verified. The randomized reference, with the same 74 held-out treated units and 260 experimental controls, was 1,826 earnings units (95% Welch interval −162 to 3,814). Observational estimates were −8,563 naive, −98 standardization, 1,167 ATT weighting, 1,277 matching and 790 ATT AIPW (approximate 95% interval −1,192 to 2,773). Matching retained all 74 treated units with 53 unique controls. The observational ATE sensitivity subset in [0.05, 0.95] retained only 269 of 6,471 rows. These sample contrasts depend on population comparability and overlap assumptions. [Publisher cohorts](https://users.nber.org/~rdehejia/nswdata.html)

## Outputs and next session

The [artifact directory](../artifacts/session_09/) includes ATE and ATT tables, matching and ESS diagnostics, overlap sensitivity, the love plot, oracle simulations, external estimates and run provenance. See [methods and reproduction](OBSERVATIONAL_ESTIMATORS.md). Session 10 can start from the observed lack of overlap, nuisance sensitivity and hidden-confounding examples. The Session 07 0/30 DML interval coverage result remains documented.
