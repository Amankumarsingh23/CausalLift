# Datasets

## Criteo Uplift Prediction Dataset v2.1

Criteo assembled the dataset from advertising incrementality experiments that randomly withheld advertising eligibility from some users. Assignment is `treatment` (1 eligible, 0 control). Use it for intention-to-treat comparisons. `exposure` records actual advertising exposure and is post-assignment. `conversion` is the primary binary outcome and `visit` is secondary. [Criteo AI Lab release and erratum](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).

Credit Eustache Diemert, Artem Betlei, Christophe Renaudin and Massih-Reza Amini, “A Large Scale Benchmark for Uplift Modeling,” AdKDD 2018. The release carries CC BY-NC-SA 4.0. Redistribution is subject to attribution, noncommercial use and ShareAlike for adaptations. Mark modifications when sharing and retain the supplied notices. Raw and derived row-level data are gitignored. [Publisher terms](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) · [License text](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The original release had advertiser-related leakage. The revised release addresses that issue. The publisher used nonuniform privacy subsampling and anonymized, projected covariates. Released-sample outcome differences cannot recover original campaign incrementality. The archive also lacks advertiser, trial and customer identifiers, so trial-specific effects and customer-level grouping cannot be checked. The released treatment share is empirical. The contract leaves known assignment probability unset.

The downloaded CSV header contains `f0`–`f11`, `treatment`, `conversion`, `visit`, and `exposure`. The detailed publisher field list and archive agree on 12 features. Parquet stores features as float64, binary fields as int8 and the persisted split as a string. Identical feature rows are retained.

The historical download link redirects to `criteostorage.blob.core.windows.net`, which failed DNS resolution locally. Criteo’s official Hugging Face archive was downloaded and matched SHA-256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`. For manual acquisition, save the publisher file as `data/raw/criteo-research-uplift-v2.1.csv.gz` or pass its path via `--dataset`. Criteo-compatible CSVs with other checksums are marked unverified in the report. [Publisher archive and SHA-256](https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz).

The build validates all rows in chunks and records full-source counts, missing and invalid values, and per-arm rates. It uses proportional quotas over joint assignment/conversion strata and uniform random-priority sampling within each stratum. The working Parquet contains only the sample. Raw input is read-only during processing.

## Synthetic datasets and fixture

Scenarios A–F generate binary assignment and outcome with known ground truth. See [CAUSAL_SIMULATOR.md](CAUSAL_SIMULATOR.md). The handmade CSV under `tests/fixtures/` tests the ingestion schema and is labeled `fixture` in test runs.
