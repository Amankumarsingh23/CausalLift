"""Cross-fitted partially linear DML and optional EconML heterogeneous models."""

import warnings
from dataclasses import dataclass
from functools import partial

import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from causallift.crossfit import crossfit_outcome
from causallift.experimentation import validated_rows
from causallift.meta_learners import fit_binary_outcome, outcome_configuration
from causallift.prediction import evaluate


@dataclass(frozen=True)
class StructuralEstimate:
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    denominator: float
    derivative: float
    score_mean: float


def orthogonal_score(y_residual, t_residual, theta):
    y, t = np.asarray(y_residual, dtype=float), np.asarray(t_residual, dtype=float)
    if y.ndim != 1 or y.shape != t.shape or len(y) < 2 or not np.isfinite([y, t]).all():
        raise ValueError("Residuals must be finite aligned vectors with at least two entries")
    return t * (y - theta * t)


def structural_effect(y_residual, t_residual) -> StructuralEstimate:
    t = np.asarray(t_residual, dtype=float)
    numerator = orthogonal_score(y_residual, t, 0)
    derivative = float(np.mean(t**2))
    if derivative <= 1e-10:
        raise ValueError("Insufficient residual treatment variation")
    theta = float(numerator.mean() / derivative)
    score = orthogonal_score(y_residual, t, theta)
    se = float(np.sqrt(np.mean(score**2) / len(t)) / derivative)
    return StructuralEstimate(theta, se, theta - 1.95996398454 * se, theta + 1.95996398454 * se,
                              derivative * len(t), -derivative, float(score.mean()))


def _continuous_model(X, outcome, *, seed, estimators):
    config = outcome_configuration("lightgbm", seed, estimators) | {"objective": "regression"}
    return lgb.LGBMRegressor(**config).fit(X, outcome)


def partially_linear_dml(X, treatment, outcome, *, propensity=None, binary_outcome=True,
                         groups=None, seed=42, estimators=65):
    """Known scalar propensity or separately cross-fitted probability model."""
    assignment, y = validated_rows(treatment, outcome, binary=binary_outcome)
    if not assignment.any() or assignment.all():
        raise ValueError("Both treatment arms are required")
    config = outcome_configuration("logistic" if binary_outcome else "lightgbm", seed, estimators)
    kwargs = {"groups": groups, "seed": seed}
    if binary_outcome:
        fit = partial(fit_binary_outcome, configuration=config, scaled_columns=tuple(X.columns))
        cross_y = crossfit_outcome(X, assignment, y, fit, **kwargs)
    else:
        fit = partial(_continuous_model, seed=seed, estimators=estimators)
        cross_y = crossfit_outcome(X, assignment, y, fit, **kwargs, binary_outcome=False,
                                   predict_outcome=lambda model, held: model.predict(held))

    if propensity is None:
        fit_t = partial(fit_binary_outcome, configuration=outcome_configuration("lightgbm", seed, estimators),
                        scaled_columns=tuple(X.columns))
        cross_t = crossfit_outcome(X, assignment, assignment, fit_t, **kwargs)
        e = cross_t.predictions
        t_diagnostics = {"strategy": cross_t.strategy, "folds": cross_t.folds,
                         "warnings": cross_t.warnings, "metrics": evaluate(assignment, e)}
    else:
        e = np.full(len(y), propensity, dtype=float)
        t_diagnostics = None
    if not np.all((0 < e) & (e < 1)):
        raise ValueError("DML requires propensity probabilities strictly between zero and one")
    yr, tr = y - cross_y.predictions, assignment.astype(float) - e
    metrics = (evaluate(y, cross_y.predictions) if binary_outcome
               else {"rmse": float(np.sqrt(np.mean(yr**2)))})
    details = {"rows": len(y), "outcome_strategy": cross_y.strategy, "outcome_folds": cross_y.folds,
               "outcome_warnings": cross_y.warnings, "outcome_metrics": metrics,
               "propensity_crossfit": t_diagnostics, "propensity_min": float(e.min()),
               "propensity_max": float(e.max()), "propensity_p10": float(np.quantile(e, 0.1)),
               "propensity_p90": float(np.quantile(e, 0.9)), "outcome_residual_mean": float(yr.mean()),
               "outcome_residual_sd": float(yr.std()), "treatment_residual_mean": float(tr.mean()),
               "treatment_residual_sd": float(tr.std())}
    return structural_effect(yr, tr), details


class FixedProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable known-design nuisance exposing positive-class probabilities."""

    def __init__(self, propensity=0.5):
        self.propensity = propensity

    def fit(self, X, y):
        if not 0 < self.propensity < 1:
            raise ValueError("Propensity must lie strictly in (0, 1)")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.tile([1 - self.propensity, self.propensity], (len(X), 1))

    def predict(self, X):
        return np.full(len(X), int(self.propensity >= 0.5))


class AdvancedDML:
    def __init__(self, kind, propensity, *, seed=42, trees=100, min_leaf=35):
        if kind not in {"linear_dml", "causal_forest"}:
            raise ValueError("Choose linear_dml or causal_forest")
        self.kind, self.propensity, self.seed, self.trees, self.min_leaf = kind, propensity, seed, trees, min_leaf

    def fit(self, X, treatment, outcome):
        from econml.dml import CausalForestDML, LinearDML

        t, y = validated_rows(treatment, outcome, binary=True)
        if len(X) != len(y) or np.unique(t).size != 2 or np.unique(y).size != 2:
            raise ValueError("Expected aligned features and both treatment and outcome classes")
        joint = 2 * t.astype(int) + y
        if np.unique(joint, return_counts=True)[1].min() < 3:
            joint = t
        folds = list(StratifiedKFold(3, shuffle=True, random_state=self.seed).split(X, joint))
        if any(np.unique(y[train]).size < 2 or np.unique(t[train]).size < 2 for train, _ in folds):
            raise ValueError("A nuisance training fold lacks an outcome or treatment class")
        model_y = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=self.seed))
        args = dict(model_y=model_y, model_t=FixedProbabilityClassifier(self.propensity),
                    discrete_treatment=True, discrete_outcome=True, cv=folds, random_state=self.seed)
        if self.kind == "linear_dml":
            self.model_ = LinearDML(**args)
        else:
            self.model_ = CausalForestDML(**args, n_estimators=self.trees, min_samples_leaf=self.min_leaf,
                                          max_depth=12, max_samples=0.4, honest=True,
                                          inference=True, n_jobs=2, drate=False)
        self.feature_names_ = tuple(X.columns)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter("always")
            self.model_.fit(y, t.astype(int), X=X.to_numpy())
        self.fit_warnings_ = tuple(dict.fromkeys(str(item.message) for item in caught))
        return self

    def _input(self, X):
        if tuple(X.columns) != self.feature_names_:
            raise ValueError("Use the original feature order")
        a = X.to_numpy()
        if not np.isfinite(a).all():
            raise ValueError("Features must be finite")
        return a

    def predict_effect(self, X):
        effect = np.asarray(self.model_.effect(self._input(X))).reshape(-1)
        if effect.shape != (len(X),) or not np.isfinite(effect).all():
            raise ValueError("Invalid effect predictions")
        return effect

    def effect_interval(self, X):
        result = self.model_.effect_inference(self._input(X))
        lo, hi = result.conf_int(alpha=0.05)
        stderr = np.asarray(result.stderr).reshape(-1)
        # EconML can return point estimates as both limits when its stderr is undefined.
        supported = np.isfinite(stderr) & (stderr > 0)
        return np.where(supported, np.asarray(lo).reshape(-1), np.nan), np.where(
            supported, np.asarray(hi).reshape(-1), np.nan,
        )

    def average_interval(self, X):
        a = self._input(X)
        lo, hi = self.model_.ate_interval(a, alpha=0.05)
        return float(np.asarray(lo).item()), float(np.asarray(hi).item())
