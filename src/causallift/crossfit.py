"""Deterministic out-of-fold nuisance predictions on a training population."""

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from causallift.experimentation import validated_rows
from causallift.meta_learners import outcome_probability


@dataclass(frozen=True)
class CrossFit:
    predictions: np.ndarray
    membership: np.ndarray
    models: tuple
    folds: tuple[dict, ...]
    strategy: str
    warnings: tuple[str, ...]


def crossfit_outcome(X: pd.DataFrame, treatment, outcome, fit_outcome, *, groups=None,
                     folds: int = 3, seed: int = 42, binary_outcome: bool = True,
                     predict_outcome=outcome_probability) -> CrossFit:
    """Fit a new outcome model per fold and predict only its held-out rows."""
    assignment, y = validated_rows(treatment, outcome, binary=binary_outcome)
    if len(X) != len(y) or not isinstance(folds, Integral) or not 2 <= folds <= len(y):
        raise ValueError("Cross-fitting requires aligned inputs and between 2 and n folds")
    group_codes = None
    if groups is not None:
        group_array = np.asarray(groups)
        if group_array.shape != y.shape or pd.isna(group_array).any():
            raise ValueError("Group identifiers must be complete and aligned")
        codes, levels = pd.factorize(group_array, sort=False)
        if len(levels) < folds:
            raise ValueError("Too few distinct groups for the requested folds")
        if len(levels) != len(y):
            group_codes = codes

    def has_support(labels):
        if group_codes is None:
            return min(np.unique(labels, return_counts=True)[1]) >= folds
        return min(len(np.unique(group_codes[labels == label])) for label in np.unique(labels)) >= folds

    warnings = []
    joint = 2 * assignment.astype(int) + y
    if binary_outcome and has_support(joint):
        labels, stratification = joint, "joint"
    elif has_support(assignment):
        labels, stratification = assignment, "treatment"
        if binary_outcome:
            warnings.append("Rare treatment/outcome cells. Stratified on treatment only.")
    else:
        labels, stratification = np.zeros(len(y), dtype=int), "unstratified"
        warnings.append("Rare treatment arms. No treatment stratification feasible.")
    if group_codes is None:
        # A single dummy stratum gives an ordinary shuffled split when stratification is infeasible.
        split = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, labels)
    elif stratification == "unstratified":
        shuffled_codes = np.random.default_rng(seed).permutation(len(levels))[group_codes]
        split = GroupKFold(folds).split(X, groups=shuffled_codes)
    else:
        split = StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, labels, group_codes)
    membership = np.full(len(y), -1, dtype=int)
    predictions = np.full(len(y), np.nan)
    models, details = [], []

    def counts(indices):
        return {"n": len(indices), "treated_n": int(assignment[indices].sum()),
                "control_n": int((~assignment[indices]).sum()),
                "treated_events": int(y[indices][assignment[indices]].sum()) if binary_outcome else None,
                "control_events": int(y[indices][~assignment[indices]].sum()) if binary_outcome else None}

    for fold, (train, held) in enumerate(split):
        if (not len(train) or not len(held) or (membership[held] != -1).any() or np.intersect1d(train, held).size
                or (group_codes is not None and np.intersect1d(group_codes[train], group_codes[held]).size)):
            raise RuntimeError("Cross-fit fold overlaps or is empty")
        start = perf_counter()
        model = fit_outcome(X.iloc[train], y[train])
        fit_seconds = perf_counter() - start
        predicted = np.asarray(predict_outcome(model, X.iloc[held]), dtype=float)
        if predicted.shape != (len(held),) or not np.isfinite(predicted).all():
            raise ValueError("Expected one finite nuisance prediction per held-out row")
        predictions[held] = predicted
        membership[held] = fold
        models.append(model)
        details.append({"fold": fold, "training": counts(train), "held_out": counts(held),
                        "constant_fallback": model if isinstance(model, float) else None,
                        "fit_seconds": fit_seconds})
    if (membership < 0).any() or not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete out-of-fold predictions")
    strategy = ("group_" if group_codes is not None else "") + stratification
    return CrossFit(predictions, membership, tuple(models), tuple(details), strategy, tuple(warnings))
