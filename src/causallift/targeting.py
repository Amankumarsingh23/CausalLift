"""Equal-capacity policy rankings and separate synthetic-only truth evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

from causallift.data.contract import DatasetSpec
from causallift.experimentation import analyze_binary

FRACTIONS = (0.10, 0.20, 0.50)


def rank_scores(scores, *, seed: int = 42) -> np.ndarray:
    """Rank descending; use a seeded common permutation to resolve equal scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Ranking requires a nonempty vector of finite scores")
    order = np.random.default_rng(seed).permutation(len(scores))
    return order[np.argsort(-scores[order], kind="stable")]


def policy_rankings(probability, train: pd.DataFrame, spec: DatasetSpec, *, seed: int = 42) -> dict:
    if spec.source_type not in {"experiment", "synthetic_randomized", "fixture"}:
        raise ValueError("The constant-effect baseline requires randomized assignment or a labelled fixture")
    if not train["split"].eq("train").all():
        raise ValueError("Constant treatment effect must use training rows")
    p = np.asarray(probability, dtype=float)
    if not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Conversion scores must be in [0, 1]")
    effect = analyze_binary(train[spec.treatment_column], train[spec.outcome_column]).absolute_lift
    random_order = rank_scores(np.zeros(p.size), seed=seed)
    return {
        "random": random_order,
        "predictive": rank_scores(p, seed=seed),
        "constant_effect": rank_scores(np.full(p.size, effect), seed=seed),
    }, effect


def _summarize(rankings: dict, probability, validation: pd.DataFrame, spec: DatasetSpec, truth=None) -> list[dict]:
    p = np.asarray(probability, dtype=float)
    if (p.shape != (len(validation),) or not np.all((0 <= p) & (p <= 1))
            or not validation["split"].eq("validation").all()):
        raise ValueError("Policy evaluation requires matching validation rows and scores")
    for policy, order in rankings.items():
        if (order.shape != p.shape or order.dtype.kind not in "iu"
                or np.any((order < 0) | (order >= len(p)))
                or not np.all(np.bincount(order.astype(np.intp, copy=False), minlength=len(p)) == 1)):
            raise ValueError(f"{policy} must rank each eligible row exactly once")
    rows = []
    for fraction in FRACTIONS:
        capacity = int(np.floor(fraction * len(p)))
        if not capacity:
            raise ValueError("Validation population is too small for the specified targeting fractions")
        for policy, order in rankings.items():
            selected = order[:capacity]
            scores = p[selected]
            observed = validation.iloc[selected]
            row = {
                "policy": policy, "fraction": fraction, "eligible_n": len(p), "selected_n": capacity,
                "mean_predicted_conversion": float(scores.mean()), "p10_predicted_conversion": float(np.quantile(scores, 0.1)),
                "p90_predicted_conversion": float(np.quantile(scores, 0.9)),
                "observed_conversion_rate": float(observed[spec.outcome_column].mean()),
                "observed_treated_share": float(observed[spec.treatment_column].mean()),
                "mean_true_cate": None, "expected_incremental_conversions": None, "negative_effect_fraction": None,
            }
            if truth is not None:
                tau = truth[selected]
                row.update(mean_true_cate=float(tau.mean()), expected_incremental_conversions=float(tau.sum()),
                           negative_effect_fraction=float((tau < 0).mean()))
            rows.append(row)
    return rows


def describe_targeting(rankings: dict, probability, validation: pd.DataFrame, spec: DatasetSpec) -> list[dict]:
    """Describe selected groups in logged data without counterfactual outcomes."""
    return _summarize(rankings, probability, validation, spec)


def aligned_synthetic_effects(validation: pd.DataFrame, spec: DatasetSpec, truth_path: Path) -> np.ndarray:
    """Read evaluation-only truth in the order of the supplied synthetic rows."""
    if spec.source_type != "synthetic_randomized" or spec.unit_id is None:
        raise ValueError("Oracle evaluation requires randomized synthetic data with unit identifiers")
    truth = pd.read_parquet(truth_path, columns=[spec.unit_id, "tau_true"])
    ordered = validation[[spec.unit_id]].merge(truth, on=spec.unit_id, how="left", validate="one_to_one")
    tau = ordered["tau_true"].to_numpy(dtype=float)
    if not np.isfinite(tau).all() or (np.abs(tau) > 1).any():
        raise ValueError("Synthetic truth is missing or invalid for the validation population")
    return tau


def evaluate_synthetic_targeting(
    rankings: dict, probability, validation: pd.DataFrame, spec: DatasetSpec, truth_path: Path, *, seed: int = 42,
) -> list[dict]:
    """Load truth only after fitting, and use it only for evaluation and oracle ranking."""
    tau = aligned_synthetic_effects(validation, spec, truth_path)
    oracle = rank_scores(tau, seed=seed)
    return _summarize(rankings | {"oracle": oracle}, probability, validation, spec, truth=tau)
