"""Exploratory binary contrasts for pre-specified subgroups."""

import numpy as np
import pandas as pd

from causallift.data.contract import DatasetSpec
from causallift.experimentation import binary_from_counts, validated_rows


def subgroup_analysis(
    frame: pd.DataFrame, spec: DatasetSpec, feature: str, *,
    bins: tuple[float, ...] | None = None, confidence_level: float = 0.95,
) -> list[dict]:
    if feature not in spec.feature_columns:
        raise ValueError(f"Subgroup feature must be allowlisted pre-treatment data: {feature}")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be strictly between zero and one")
    assigned, outcome = validated_rows(frame[spec.treatment_column], frame[spec.outcome_column], binary=True)
    values = frame[feature]
    if values.isna().any():
        raise ValueError("Subgroup feature contains missing values")
    groups = values
    if bins is not None:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(edges[1:] > edges[:-1]):
            raise ValueError("Bin boundaries must be strictly increasing")
        groups = pd.cut(values, edges, include_lowest=True)
        if groups.isna().any():
            raise ValueError("Bin boundaries must cover all subgroup values")
    rows = pd.DataFrame({"group": groups.to_numpy() if bins is None else groups.array,
                         "assigned": assigned, "outcome": outcome})
    result = []
    for group, sample in rows.groupby("group", observed=False, sort=True):
        treated = sample.loc[sample["assigned"], "outcome"]
        control = sample.loc[~sample["assigned"], "outcome"]
        row = {
            "dataset": spec.dataset_id, "feature": feature, "subgroup": str(group), "exploratory": True,
            "treated_n": len(treated), "control_n": len(control),
            "treated_events": int(treated.sum()), "control_events": int(control.sum()),
            "treated_rate": float(treated.mean()) if len(treated) else None,
            "control_rate": float(control.mean()) if len(control) else None,
            "absolute_lift": None, "ci_low": None, "ci_high": None, "p_value": None, "test_method": None,
        }
        if treated.empty or control.empty:
            row["warnings"] = "Insufficient data: one or both arms absent."
        else:
            test = binary_from_counts(int(treated.sum()), len(treated), int(control.sum()), len(control),
                                      confidence_level=confidence_level)
            row.update(absolute_lift=test.absolute_lift, ci_low=test.confidence_interval[0],
                       ci_high=test.confidence_interval[1], p_value=test.p_value,
                       test_method=test.test_method, warnings=" | ".join(test.warnings))
        result.append(row)
    return result
