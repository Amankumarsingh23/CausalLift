"""Prespecified continuous-outcome demonstration with observed confounding."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from causallift.double_ml import partially_linear_dml
from causallift.experimentation import analyze_continuous

TRUE_EFFECT = 1.25
FEATURES = tuple(f"x{i:02d}" for i in range(30))


def simulate_confounding(seed, rows=2400):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, len(FEATURES)))
    nonlinear = np.column_stack([x[:, 0] ** 2 - 1, x[:, 1] * x[:, 2], np.sin(x[:, 3])])
    index = nonlinear @ np.array([0.85, 0.85, 0.4]) + 0.4 * x[:, 4]
    from scipy.special import expit

    propensity = 0.10 + 0.80 * expit(index)
    treatment = rng.binomial(1, propensity)
    baseline = nonlinear @ np.array([0.9, 0.9, 0.45]) + 0.50 * x[:, 4] + 0.15 * x[:, 5]
    outcome = baseline + TRUE_EFFECT * treatment + rng.normal(size=rows)
    return pd.DataFrame(x, columns=FEATURES), treatment, outcome, propensity


def _linear_adjustment(X, t, y):
    design = np.column_stack([np.ones(len(y)), t, X])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1] or len(y) <= rank:
        raise ValueError("Linear adjustment requires a full-rank design and residual degrees of freedom")
    projection = np.linalg.solve(design.T @ design, design.T)[1]
    error = y - design @ beta
    se = float(np.sqrt(np.sum((projection * error) ** 2) * len(y) / (len(y) - rank)))
    return float(beta[1]), se, float(beta[1] - 1.95996398454 * se), float(beta[1] + 1.95996398454 * se)


def confounding_study(seed=42, repetitions=30, rows=2400, estimators=55):
    if repetitions < 1 or rows < len(FEATURES) + 3:
        raise ValueError("Confounding study requires repetitions and enough rows for adjustment")
    results, diagnostics = [], []
    for rep, child_seed in enumerate(np.random.SeedSequence(seed).spawn(repetitions)):
        sample_seed = int(child_seed.generate_state(1)[0])
        X, t, y, true_e = simulate_confounding(sample_seed, rows)
        naive = analyze_continuous(t, y)
        linear = _linear_adjustment(X, t, y)
        dml, detail = partially_linear_dml(X, t, y, binary_outcome=False, seed=sample_seed,
                                            estimators=estimators)
        diagnostics.append({"rep": rep, "seed": sample_seed, **asdict(dml), **detail})
        for method, (point, se, lo, hi) in {
            "naive": (naive.absolute_lift, naive.effect_se, *naive.confidence_interval),
            "regression_adjustment": linear,
            "partially_linear_dml": (dml.estimate, dml.standard_error, dml.ci_low, dml.ci_high),
        }.items():
            results.append({"rep": rep, "seed": sample_seed, "method": method, "rows": rows,
                            "true_effect": TRUE_EFFECT, "estimate": point, "standard_error": se,
                            "ci_low": lo, "ci_high": hi, "signed_error": point - TRUE_EFFECT,
                            "covers": lo <= TRUE_EFFECT <= hi, "true_propensity_min": float(true_e.min()),
                            "true_propensity_max": float(true_e.max()), "treated_share": float(t.mean()),
                            "estimated_propensity_p10": detail["propensity_p10"] if method == "partially_linear_dml" else None,
                            "estimated_propensity_p90": detail["propensity_p90"] if method == "partially_linear_dml" else None})
    frame = pd.DataFrame(results)
    summary = {key: {"average_estimate": float(g.estimate.mean()), "average_bias": float(g.signed_error.mean()),
                     "rmse": float(np.sqrt(np.mean(g.signed_error**2))), "coverage": float(g.covers.mean()),
                     "repetitions": len(g)}
               for key, g in frame.groupby("method")}
    return results, summary, diagnostics
