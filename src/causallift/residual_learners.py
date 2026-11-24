"""X-learning and residualized R-learning with a constant design propensity."""

import hashlib
from dataclasses import dataclass
from functools import partial

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from causallift.crossfit import crossfit_outcome
from causallift.data.contract import DatasetSpec
from causallift.experimentation import validated_rows
from causallift.meta_learners import (
    MetaLearner,
    fit_binary_outcome,
    outcome_configuration,
    outcome_probability,
)
from causallift.prediction import evaluate

MIN_RESIDUAL = 1e-8


@dataclass(frozen=True)
class ConstantPropensity:
    value: float
    source: str

    def __post_init__(self):
        if not 0 < self.value < 1:
            raise ValueError("Propensity must lie strictly in (0, 1)")

    def predict(self, X):
        return np.full(len(X), self.value)


def training_propensity(spec: DatasetSpec, train) -> ConstantPropensity:
    if "split" not in train or train.empty or not train["split"].eq("train").all():
        raise ValueError("Propensity requires training rows only")
    if spec.source_type == "synthetic_randomized" and spec.assignment_probability is not None:
        return ConstantPropensity(spec.assignment_probability, "known randomized design")
    if spec.source_type not in {"experiment", "fixture"}:
        raise ValueError("Training propensity requires randomized design or the released-benchmark approximation")
    t = train[spec.treatment_column]
    if not t.isin([0, 1]).all():
        raise ValueError("Training assignment must be binary")
    return ConstantPropensity(float(t.mean()), "released training share" if spec.source_type == "experiment"
                              else "fixture training share")


def x_pseudo_effects(assignment, outcome, mu0, mu1) -> np.ndarray:
    t, y = validated_rows(assignment, outcome, binary=True)
    p0, p1 = np.asarray(mu0), np.asarray(mu1)
    if any(p.shape != y.shape or not np.all((0 <= p) & (p <= 1)) for p in (p0, p1)):
        raise ValueError("Potential outcomes must be aligned probabilities")
    return np.where(t, y - p0, p1 - y)


def residuals(treatment, outcome, nuisance, propensity) -> tuple[np.ndarray, np.ndarray]:
    t, y = validated_rows(treatment, outcome, binary=True)
    m, e = np.asarray(nuisance, dtype=float), np.broadcast_to(np.asarray(propensity, dtype=float), y.shape)
    if m.shape != y.shape or not np.all((0 <= m) & (m <= 1)):
        raise ValueError("Nuisance predictions must be aligned probabilities")
    if not np.all((0 < e) & (e < 1)):
        raise ValueError("Propensities must lie strictly in (0, 1)")
    return y - m, t.astype(float) - e


def weighted_r_targets(y_residual, treatment_residual) -> tuple[np.ndarray, np.ndarray]:
    y_res, t_res = np.asarray(y_residual, dtype=float), np.asarray(treatment_residual, dtype=float)
    if (y_res.ndim != 1 or not y_res.size or y_res.shape != t_res.shape
            or not np.isfinite(y_res).all() or not np.isfinite(t_res).all()):
        raise ValueError("R-loss needs aligned finite residuals")
    if (np.abs(t_res) < MIN_RESIDUAL).any():
        raise ValueError("A treatment residual is too close to zero for weighted regression")
    return y_res / t_res, t_res**2


def effect_configuration(base: str, seed: int, estimators: int) -> dict:
    if base == "linear":
        return {"alpha": 1.0}
    return outcome_configuration("lightgbm", seed, estimators) | {"objective": "regression"}


def _fit_effect(X, pseudo, base, config, weight=None):
    if base == "linear":
        model = make_pipeline(StandardScaler(), Ridge(**config))
        kwargs = {"ridge__sample_weight": weight} if weight is not None else {}
        with threadpool_limits(limits=2):
            return model.fit(X, pseudo, **kwargs)
    return lgb.LGBMRegressor(**config).fit(X, pseudo, sample_weight=weight)


class ResidualLearner:
    """Effect-only X/R learner; fit inputs must come from the training partition."""

    def __init__(self, learner: str, base: str, propensity: ConstantPropensity, *, seed: int = 42,
                 estimators: int = 65, folds: int = 3):
        if learner not in {"x", "r"} or base not in {"linear", "lightgbm"} or estimators < 1:
            raise ValueError("Use an X/R learner and linear/LightGBM base with a positive tree limit")
        self.learner, self.base, self.propensity = learner, base, propensity
        self.seed, self.estimators, self.folds = seed, estimators, folds

    @property
    def model_config(self) -> dict:
        base = "logistic" if self.base == "linear" else "lightgbm"
        return {"outcome": outcome_configuration(base, self.seed, self.estimators),
                "effect": effect_configuration(self.base, self.seed, self.estimators), "folds": self.folds}

    def fit(self, X, treatment, outcome, *, groups=None):
        t, y = validated_rows(treatment, outcome, binary=True)
        e = self.propensity.predict(X)
        config = self.model_config
        if self.learner == "x":
            baseline = MetaLearner("t", "logistic" if self.base == "linear" else "lightgbm",
                                   seed=self.seed, estimators=self.estimators).fit(X, t, y)
            # Impute each arm only with its opposite-arm model.
            p1_control = outcome_probability(baseline.models_[1], X.loc[~t])
            p0_treated = outcome_probability(baseline.models_[0], X.loc[t])
            p0 = np.zeros(len(y))
            p1 = np.zeros(len(y))
            p0[t], p1[~t] = p0_treated, p1_control
            d = x_pseudo_effects(t, y, p0, p1)
            self.effect_models_ = tuple(_fit_effect(X.loc[mask], d[mask], self.base, config["effect"])
                                        for mask in (~t, t))
            self.diagnostics_ = {"outcome_fallbacks": baseline.fallbacks_,
                                 "control_pseudo_mean": float(d[~t].mean()), "treated_pseudo_mean": float(d[t].mean())}
        else:
            # Validate the same strict feature matrix and assignment contract used by S/T.
            MetaLearner("t", "logistic")._features(X, fitted=False)
            if len(X) != len(y) or not t.any() or t.all():
                raise ValueError("Aligned rows with both treatment arms are required")
            fit = partial(fit_binary_outcome, configuration=config["outcome"], scaled_columns=tuple(X.columns))
            cross = crossfit_outcome(X, t, y, fit, groups=groups, folds=self.folds, seed=self.seed)
            yr, tr = residuals(t, y, cross.predictions, e)
            target, weight = weighted_r_targets(yr, tr)
            self.effect_models_ = (_fit_effect(X, target, self.base, config["effect"], weight / weight.mean()),)
            self.nuisance_models_ = cross.models
            self.diagnostics_ = {
                "strategy": cross.strategy, "warnings": list(cross.warnings), "folds": list(cross.folds),
                "covered_rows": len(cross.predictions),
                "membership_sha256": hashlib.sha256(cross.membership.astype("<i4").tobytes()).hexdigest(),
                "oof_metrics": evaluate(y, cross.predictions),
                "propensity_min": float(e.min()), "propensity_max": float(e.max()),
                "outcome_residual_mean": float(yr.mean()), "outcome_residual_sd": float(yr.std()),
                "treatment_residual_min": float(tr.min()), "treatment_residual_max": float(tr.max()),
                "treatment_residual_mean": float(tr.mean()), "treatment_residual_sd": float(tr.std()),
                "minimum_absolute_treatment_residual": float(np.abs(tr).min()),
                "mean_raw_weight": float(weight.mean()),
            }
        self.feature_columns_ = tuple(X.columns)
        return self

    def _check_features(self, X):
        MetaLearner("t", "logistic")._features(X, fitted=False)
        if not hasattr(self, "feature_columns_") or tuple(X.columns) != self.feature_columns_:
            raise ValueError("Predict with the fitted feature names in their original order")

    def predict_effect(self, X):
        self._check_features(X)
        if self.learner == "x":
            e = self.propensity.predict(X)
            effect = e * self.effect_models_[0].predict(X) + (1 - e) * self.effect_models_[1].predict(X)
        else:
            effect = self.effect_models_[0].predict(X)
        if not np.isfinite(effect).all():
            raise ValueError("Nonfinite effect predictions")
        return effect

    def validation_r_loss(self, X, treatment, outcome, effect=None):
        if self.learner != "r":
            raise ValueError("Validation R-loss requires a fitted R-learner")
        self._check_features(X)
        m = np.mean([outcome_probability(model, X) for model in self.nuisance_models_], axis=0)
        yr, tr = residuals(treatment, outcome, m, self.propensity.predict(X))
        prediction = self.predict_effect(X) if effect is None else np.asarray(effect)
        if prediction.shape != yr.shape or not np.isfinite(prediction).all():
            raise ValueError("R-loss requires aligned finite effects")
        return float(np.mean((yr - tr * prediction) ** 2))
