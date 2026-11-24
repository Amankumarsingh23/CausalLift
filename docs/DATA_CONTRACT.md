# Causal data contract

`DatasetSpec` carries dataset identity, source type, assignment, primary and secondary outcomes, optional unit ID and assignment probability, predictor allowlist, exclusions, truth columns, and provenance. Each observed Parquet has a sibling `.spec.json`.

Get the design matrix through `spec.features(frame)`. The allowlist excludes assignment, all outcomes, post-treatment variables, identifiers, persisted splits, explicitly excluded fields and truth. Missing required values, unsupported predictor dtypes, nonfinite values, invalid binary values and missing arms are rejected. The report records missing and invalid counts for successful runs. Validation errors include affected columns. Unparseable CSV values fail during parsing.

Load a partition with `load_dataset(path, spec, "train")`. For Criteo use `causallift.data.criteo.CRITEO`. For synthetic data, `simulate("A")[2]` returns the scenario A spec. Join synthetic truth by `unit_id` within a single scenario for evaluation.

## Partition methodology

Independent rows are shuffled within joint assignment/primary outcome strata with the specified seed. Largest-remainder allocation approximates 60/20/20 and keeps strata of at least three rows represented where feasible. When a joint allocation cannot cover both arms, the splitter warns and uses assignment strata. Missing joint cells are reported via warnings. Impossible arm coverage raises an error.

If unit IDs repeat, the splitter evaluates deterministic five-fold stratified group assignments, combining three folds for training and selecting held-out folds by deviation from target joint counts. Group integrity is enforced throughout. Group size and rare strata can limit balance. Criteo has no IDs and uses independent row assignment. One split label per Parquet row persists membership. Split reports include per-arm counts, proportions and outcome rates. Any fitted preprocessing belongs on train, model selection on validation, and final evaluation on test.
