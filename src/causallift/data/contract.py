from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    source_type: Literal["experiment", "synthetic_randomized", "synthetic_observational", "fixture"]
    treatment_column: str
    outcome_column: str
    feature_columns: tuple[str, ...]
    unit_id: str | None = None
    assignment_probability: float | None = None
    ground_truth_columns: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()
    secondary_outcomes: tuple[str, ...] = ()
    post_treatment_columns: tuple[str, ...] = ()
    provenance: dict[str, str] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    @property
    def binary_columns(self) -> tuple[str, ...]:
        return (self.treatment_column, self.outcome_column, *self.secondary_outcomes,
                *self.post_treatment_columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (*self.feature_columns, *self.binary_columns,
                *((self.unit_id,) if self.unit_id else ()))

    def __post_init__(self):
        reserved = {
            *self.binary_columns, *self.ground_truth_columns, *self.excluded_columns,
            self.unit_id, "split",
        }
        overlap = reserved.intersection(self.feature_columns)
        if overlap:
            raise ValueError(f"Forbidden model features: {sorted(overlap)}")
        if not self.feature_columns or len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("Feature and observed column roles must be present and unique")
        if self.assignment_probability is not None and not 0 < self.assignment_probability < 1:
            raise ValueError("Assignment probability must be strictly between zero and one")

    def quality(self, frame: pd.DataFrame) -> dict:
        if not frame.columns.is_unique:
            raise ValueError("Duplicate column names")
        absent = set(self.required_columns).difference(frame.columns)
        if absent:
            raise ValueError(f"Missing required columns: {sorted(absent)}")
        missing = frame.isna().sum().astype(int).to_dict()
        invalid = dict.fromkeys(frame.columns, 0)
        for name in self.binary_columns:
            values = frame[name]
            invalid[name] = int((values.notna() & ~values.isin((0, 1))).sum())
        for name in self.feature_columns:
            values = frame[name]
            if not any(check(values.dtype) for check in (is_bool_dtype, is_float_dtype, is_integer_dtype)):
                invalid[name] = int(values.notna().sum())
            else:
                invalid[name] = int((values.notna() & ~np.isfinite(values)).sum())
        return {"missing": missing, "invalid": invalid}

    def validate(self, frame: pd.DataFrame, *, require_both_arms: bool = True) -> None:
        quality = self.quality(frame)
        problems = {
            name: {kind: counts[name] for kind, counts in quality.items() if counts[name]}
            for name in self.required_columns
            if any(counts[name] for counts in quality.values())
        }
        if frame.empty or problems:
            raise ValueError(f"Invalid {self.dataset_id} data ({len(frame)} rows): {problems}")
        if require_both_arms and frame[self.treatment_column].nunique() != 2:
            raise ValueError("Both treatment arms are required")

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return only explicitly permitted predictors."""
        return frame.loc[:, list(self.feature_columns)].copy()


def load_dataset(path, spec: DatasetSpec, partition: str | None = None) -> pd.DataFrame:
    filters = [("split", "=", partition)] if partition is not None else None
    frame = pd.read_parquet(path, filters=filters)
    spec.validate(frame)
    return frame
