"""Train-only assignment probabilities and empirical overlap diagnostics."""

from functools import partial

import numpy as np

from causallift.crossfit import crossfit_outcome
from causallift.experimentation import validated_rows
from causallift.meta_learners import (
    fit_binary_outcome,
    outcome_configuration,
    outcome_probability,
)


def fit_assignments(features, assignment, name, *, seed=42, groups=None):
    t, _ = validated_rows(assignment, assignment, binary=True)
    if name not in ("constant", "logistic", "lightgbm"):
        raise ValueError("Unknown propensity model")
    config = {"model": "training assignment share"} if name == "constant" else outcome_configuration(name, seed, 55)
    fit = (constant_share if name == "constant" else
           partial(fit_binary_outcome, configuration=config, scaled_columns=tuple(features.columns)))
    oof = crossfit_outcome(features, t, t, fit, groups=groups, seed=seed)
    final = fit(features, t)
    return final, oof, config


def constant_share(features, assignment):
    return float(np.mean(assignment))


def propensity_prediction(model, features):
    return outcome_probability(model, features)


def overlap(treatment, probabilities, *, low=0.05, high=0.95):
    t, _ = validated_rows(treatment, treatment, binary=True)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != t.shape or not np.all((0 <= p) & (p <= 1)) or not 0 <= low < high <= 1:
        raise ValueError("Invalid probabilities or overlap thresholds")
    arms = [p[~t], p[t]]
    warnings = []
    if all(len(a) for a in arms):
        lower, upper = max(a.min() for a in arms), min(a.max() for a in arms)
        retained = (p >= lower) & (p <= upper)
        interval = (float(lower), float(upper)) if lower <= upper else (None, None)
        if lower > upper:
            warnings.append("Arm ranges do not intersect.")
    else:
        retained = np.zeros(len(p), dtype=bool)
        interval = (None, None)
        warnings.append("One assignment arm is absent. Common support is unavailable.")
    if np.ptp(p) == 0:
        warnings.append("Constant scores cannot diagnose covariate-level overlap.")
    if ((p < low) | (p > high)).any():
        warnings.append("Some scores are in the prespecified tails.")
    if ((p == 0) | (p == 1)).any():
        warnings.append("Exact boundary probabilities are unsafe as weighting denominators.")
    stats = {}
    for name, arm in zip(("control", "treated"), arms, strict=True):
        q = np.quantile(arm, [0, .05, .25, .5, .75, .95, 1]) if len(arm) else [None] * 7
        stats.update({f"{name}_q{pct}": None if v is None else float(v)
                      for pct, v in zip((0, 5, 25, 50, 75, 95, 100), q, strict=True)})
    return {"n": len(p), "minimum": float(p.min()), "maximum": float(p.max()),
            "support_low": interval[0], "support_high": interval[1],
            "supported_n": int(retained.sum()) if all(len(a) for a in arms) else None,
            "outside_support_fraction": float((~retained).mean()) if all(len(a) for a in arms) else None,
            "low_threshold": low, "high_threshold": high, "below_threshold": float((p < low).mean()),
            "above_threshold": float((p > high).mean()), "warnings": " ".join(warnings), **stats}
