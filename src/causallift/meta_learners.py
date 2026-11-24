"""Small S/T learners for binary outcomes and a shared effect-only interface."""

from dataclasses import dataclass
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from causallift.experimentation import validated_rows

ASSIGNMENT = "__assignment__"


class EffectEstimator(Protocol):
    def fit(self, X: pd.DataFrame, treatment, outcome) -> "EffectEstimator": ...

    def predict_effect(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class PotentialOutcomes:
    mu0: np.ndarray
    mu1: np.ndarray

    @property
    def effect(self) -> np.ndarray:
        return self.mu1 - self.mu0


def outcome_configuration(base: str, seed: int, estimators: int) -> dict:
    if base == "logistic":
        return {"C": 1.0, "solver": "lbfgs", "max_iter": 200, "tol": 1e-6, "random_state": seed}
    return {"objective": "binary", "n_estimators": estimators, "learning_rate": 0.07,
            "num_leaves": 15, "min_child_samples": 40, "reg_lambda": 1.0, "random_state": seed,
            "deterministic": True, "force_col_wise": True, "n_jobs": 2, "verbosity": -1}


def fit_binary_outcome(X: pd.DataFrame, y: np.ndarray, configuration: dict, scaled_columns: tuple[str, ...]):
    if np.unique(y).size == 1:
        return float(y[0])
    if "solver" in configuration:
        scale = ColumnTransformer([("features", StandardScaler(), list(scaled_columns))], remainder="passthrough")
        model = make_pipeline(scale, LogisticRegression(**configuration))
        with threadpool_limits(limits=2):
            return model.fit(X, y)
    return lgb.LGBMClassifier(**configuration).fit(X, y)


def outcome_probability(model, X: pd.DataFrame) -> np.ndarray:
    p = np.full(len(X), model) if isinstance(model, float) else model.predict_proba(X)[:, 1]
    if p.shape != (len(X),) or not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Outcome probabilities must be finite and in [0, 1]")
    return p


class MetaLearner:
    def __init__(self, learner: str, base: str, *, seed: int = 42, estimators: int = 100):
        if learner not in {"s", "t"} or base not in {"logistic", "lightgbm"}:
            raise ValueError("Use an S/T learner and logistic/LightGBM base")
        if estimators < 1:
            raise ValueError("The tree count must be positive")
        self.learner = learner
        self.base = base
        self.seed = seed
        self.estimators = estimators

    @property
    def model_config(self) -> dict:
        return outcome_configuration(self.base, self.seed, self.estimators)

    def _features(self, X: pd.DataFrame, *, fitted: bool) -> None:
        if (not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique
                or not all(isinstance(name, str) for name in X.columns) or ASSIGNMENT in X):
            raise ValueError("Expected a nonempty DataFrame with unique string feature names")
        if fitted and (not hasattr(self, "feature_columns_") or tuple(X.columns) != self.feature_columns_):
            raise ValueError("Predict with the fitted feature names in their original order")
        values = X.to_numpy()
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError("Features must be finite numeric values")

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray):
        return fit_binary_outcome(X, y, self.model_config, self.feature_columns_)

    def fit(self, X: pd.DataFrame, treatment, outcome) -> "MetaLearner":
        self._features(X, fitted=False)
        assigned, y = validated_rows(treatment, outcome, binary=True)
        if len(X) != len(y):
            raise ValueError("Features, treatment and outcome must have equal lengths")
        if not assigned.any() or assigned.all():
            raise ValueError("Both treatment arms are required")
        self.feature_columns_ = tuple(X.columns)
        if self.learner == "s":
            self.models_ = (self._fit_model(X.assign(**{ASSIGNMENT: assigned.astype(int)}), y),)
        else:
            self.models_ = tuple(self._fit_model(X.loc[mask], y[mask]) for mask in (~assigned, assigned))
        self.arm_stats_ = {
            name: {"n": int(mask.sum()), "positives": int(y[mask].sum())}
            for name, mask in (("control", ~assigned), ("treated", assigned))
        }
        self.fallbacks_ = {
            name: model for name, model in zip(("pooled",) if self.learner == "s" else ("control", "treated"),
                                               self.models_, strict=True)
            if isinstance(model, float)
        }
        return self

    def predict_potential_outcomes(self, X: pd.DataFrame) -> PotentialOutcomes:
        self._features(X, fitted=True)
        predictions = []
        for arm in (0, 1):
            if self.learner == "s":
                features, model = X.assign(**{ASSIGNMENT: arm}), self.models_[0]
            else:
                features, model = X, self.models_[arm]
            predictions.append(outcome_probability(model, features))
        return PotentialOutcomes(*predictions)

    def predict_effect(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_potential_outcomes(X).effect
