"""Pooled binary prediction under the logged assignment mixture."""

import hashlib
from dataclasses import dataclass
from inspect import signature

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from causallift.data.contract import DatasetSpec

MODELS = ("prevalence", "logistic", "lightgbm")
EARLY_STOPPING_ROUNDS = 15


def _binary(values) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or not y.size or y.dtype.kind not in "biuf" or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected a nonempty binary target")
    return y.astype(int, copy=False)


def _outcomes_and_scores(outcome, probability):
    y = _binary(outcome)
    p = np.asarray(probability, dtype=float)
    if p.shape != y.shape or not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Probabilities must align with outcomes and be finite in [0, 1]")
    return y, p


@dataclass
class PredictiveModel:
    name: str
    spec: DatasetSpec
    outcome: str
    estimator: object
    configuration: dict
    best_iteration: int | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        # Standard sklearn estimators initialize an array even for a single class.
        features = self.spec.features(frame)
        classes = self.estimator.classes_
        if 1 not in classes:
            return np.zeros(len(frame), dtype=float)
        probabilities = self.estimator.predict_proba(features)
        return probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def fit_model(
    name: str, spec: DatasetSpec, train: pd.DataFrame, validation: pd.DataFrame, outcome: str, *,
    seed: int = 42, estimators: int = 160,
) -> PredictiveModel:
    if outcome not in (spec.outcome_column, *spec.secondary_outcomes):
        raise ValueError("Outcome must be an explicitly configured prediction target")
    for frame, partition in ((train, "train"), (validation, "validation")):
        if "split" not in frame or not frame["split"].eq(partition).all():
            raise ValueError(f"Expected only {partition} rows")
    x_train, y_train = spec.features(train), _binary(train[outcome])
    x_val, y_val = spec.features(validation), _binary(validation[outcome])
    best = None
    if name == "prevalence":
        config = {"strategy": "prior"}
        model = DummyClassifier(**config).fit(x_train, y_train)
    elif name in ("logistic", "lightgbm"):
        if np.unique(y_train).size != 2:
            raise ValueError("Logistic regression and LightGBM require both classes in training")
        if name == "logistic":
            config = {"scaling": "StandardScaler", "C": 1.0, "solver": "lbfgs", "max_iter": 200,
                      "tol": 1e-6, "random_state": seed}
            model = make_pipeline(StandardScaler(), LogisticRegression(
                **{key: val for key, val in config.items() if key != "scaling"},
            ))
            with threadpool_limits(limits=2):
                model.fit(x_train, y_train)
        else:
            config = {
                "objective": "binary", "metric": "binary_logloss", "n_estimators": estimators,
                "learning_rate": 0.07, "num_leaves": 15, "min_child_samples": 40,
                "reg_lambda": 1.0, "random_state": seed, "deterministic": True,
                "force_col_wise": True, "n_jobs": 2, "verbosity": -1,
            }
            model = lgb.LGBMClassifier(**config)
            evaluation = ({"eval_X": x_val, "eval_y": y_val} if "eval_X" in signature(model.fit).parameters
                          else {"eval_set": [(x_val, y_val)]})
            model.fit(x_train, y_train, **evaluation,
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            config["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            best = int(model.best_iteration_)
    else:
        raise ValueError(f"Unknown predictive model: {name}")
    return PredictiveModel(name, spec, outcome, model, config, best)


def training_fingerprint(train: pd.DataFrame, spec: DatasetSpec, outcome: str) -> str:
    """Hash ordered train-only model inputs and target, including schema and dtypes."""
    frame = train.loc[:, [*spec.feature_columns, outcome]]
    digest = hashlib.sha256(repr(tuple(zip(frame.columns, map(str, frame.dtypes), strict=True))).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def evaluate(outcome, probability) -> dict:
    y, p = _outcomes_and_scores(outcome, probability)
    both_classes = np.unique(y).size == 2
    return {
        "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "mean_probability": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "warning": "" if both_classes else "Single outcome class. Ranking metrics are unavailable.",
    }


def calibration_bins(outcome, probability, *, bins: int = 10) -> list[dict]:
    y, p = _outcomes_and_scores(outcome, probability)
    if bins < 1:
        raise ValueError("Calibration requires at least one bin")
    # Interior quantile cuts keep tied scores together. Empty groups are omitted.
    cuts = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1]))
    labels = np.searchsorted(cuts, p, side="left")
    rows = []
    for index, label in enumerate(np.unique(labels), start=1):
        mask = labels == label
        rows.append({
            "bin": index, "count": int(mask.sum()), "positives": int(y[mask].sum()),
            "mean_probability": float(p[mask].mean()), "observed_rate": float(y[mask].mean()),
            "min_probability": float(p[mask].min()), "max_probability": float(p[mask].max()),
        })
    return rows
