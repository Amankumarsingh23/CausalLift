# Session 10 completion

The full benchmark ran in 38.04 seconds with seed 42. It used fixed training and validation partitions and checked saved propensity configuration, ordered fingerprints, and IDs. Test partitions remained reserved. No dependencies were added.

## Matching correction

| Cohort | Matched treated | Unique controls before → after | Maximum reuse before → after | Control ESS before → after | Estimate before → after |
| --- | ---: | --- | --- | --- | --- |
| A | 1,984 | 1 → 1,984 | 1,984 → 1 | 1 → 1,984 | −55.09 → 10.38 pp |
| C | 1,722 | 678 → 678 | 60 → 60 | 187.91 → 187.91 | 10.92 → 10.92 pp |
| F | 1,899 | 749 → 749 | 52 → 52 | 253.58 → 253.58 | 21.48 → 21.48 pp |
| NSW/CPS | 74 | 53 → 53 | 4 → 4 | 41.48 → 41.48 | 1,277.24 → 1,277.24 earnings units |

Matched truth was 10.00 pp in A, 4.12 pp in C and 5.83 pp in F. The caliper, distances and matched populations stayed fixed. Published Session 09 outputs are preserved. [Matching distances and diagnostics](../artifacts/session_10/matching_tie_correction.csv)

## Refutations

- A's 40 placebos averaged −0.518 pp (SD 1.324 pp, 5th–95th percentiles −2.136 to 1.269 pp). One nominal interval rejected zero. The original AIPW was +10.21 pp.
- C's 15 Gaussian noise refits shifted AIPW by −0.006 pp on average (SD 0.046 pp). This checks sensitivity to irrelevant covariates.
- The 20 stratified C subsets averaged −1.91 pp (SD 1.10 pp), with mean truth −0.068 pp and truth SD 0.073 pp. F subsets averaged +14.46 pp against mean truth −0.088 pp. Small variation can coexist with substantial bias.
- At zero hidden strength, the main outcome had naive 1.687, regression 1.089, and AIPW 1.083 against truth 1. The negative control had naive 0.650, regression −0.056, and AIPW 0.055 against truth zero. At strength 1.6, observed-feature negative-control AIPW was 1.725 despite zero effect.

## Hidden confounding and support

| Strength | Observed AIPW mean | Full-information AIPW mean | True propensity tails |
| --- | ---: | ---: | ---: |
| 0.0 | 0.994 | 0.994 | 0.00% |
| 0.4 | 1.157 | 0.998 | 0.03% |
| 0.8 | 1.559 | 0.978 | 0.59% |
| 1.2 | 2.126 | 0.965 | 3.45% |
| 1.6 | 2.772 | 1.008 | 9.40% |

Each strength uses 24 repetitions with true effect 1. The highest-strength observed AIPW intervals covered truth in 0/24 draws, versus 24/24 for full-information AIPW. Tail shares changed with confounding, so this is a joint sensitivity and overlap example.

In C, unmodified logistic AIPW was −1.75 pp with maximum weight 110.71. LightGBM produced +0.10 pp and maximum weight 27.79. Logistic clipping retained 4,000 rows and produced −0.63 pp. Logistic restriction retained 3,582 rows, moved true ATE to +0.39 pp, and produced AIPW +1.45 pp. Per-arm ESS and weighted SMD are in the CSV. This comparison cannot establish which real-world nuisance would be better.

A is supported under assumptions, C warrants caution for weight concentration and nuisance sensitivity, and F remains not identified. NSW/CPS is included only in the matching comparison. No negative controls were invented for real data. Session 07's 0/30 DML coverage finding remains unchanged.

## Artifacts and handoff

Run `python -m causallift.session10` or `python -m causallift.session10 --fixture`. The [artifact directory](../artifacts/session_10/) contains matching, placebo, random feature, subset, negative control, overlap and hidden sensitivity reports, two plots and a structured reliability report. [Methods](CAUSAL_RELIABILITY.md)

Session 11 can carry the identification and support flags into policy comparisons when treatment value and cost are introduced. No business decisions should depend on unsupported estimates presented as causal.
