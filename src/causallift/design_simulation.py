"""Seeded Monte Carlo checks for classical experiment designs."""

from math import sqrt

import numpy as np
from scipy.stats import norm

from causallift.cuped import analyze_cuped, fit_cuped
from causallift.experiment_health import adjust_p_values
from causallift.experimentation import binary_from_counts
from causallift.planning import binary_sample_size, validate_counts

CORRELATION_CASES = {"strong": 2.0, "weak": 0.15}
MULTIPLICITY_METHODS = ("bonferroni", "holm", "benjamini_hochberg")


def cuped_examples(seed: int, repetitions: int) -> tuple[dict, list[dict]]:
    validate_counts(repetitions, minimum=2)
    output, rows = {}, []
    n, pilot_n, effect = 500, 2000, 0.4
    for rng, (label, slope) in zip(np.random.SeedSequence(seed).spawn(2), CORRELATION_CASES.items(), strict=True):
        generator = np.random.default_rng(rng)
        pilot_x = generator.normal(size=pilot_n)
        fit = fit_cuped(pilot_x, 10 + slope * pilot_x + generator.normal(size=pilot_n))
        assignment = generator.permutation(np.repeat([0, 1], n))
        x = generator.normal(size=2 * n)
        y = 10 + slope * x + effect * assignment + generator.normal(size=2 * n)
        output[label] = analyze_cuped(assignment, y, x, fit)
        # Independent Gaussian arm-mean differences have this exact sampling distribution.
        x_diff, noise_diff = generator.normal(scale=sqrt(2 / n), size=(2, repetitions))
        raw = effect + slope * x_diff + noise_diff
        adjusted = raw - fit.theta * x_diff
        raw_var, adjusted_var = float(raw.var(ddof=1)), float(adjusted.var(ddof=1))
        reduction = 1 - adjusted_var / raw_var
        influence = ((adjusted_var / raw_var) * (raw - raw.mean()) ** 2
                     - (adjusted - adjusted.mean()) ** 2) / raw_var
        rows.append({
            "check": f"cuped_{label}", "metric": "variance_reduction", "estimate": reduction,
            "monte_carlo_se": float(influence.std(ddof=1) / sqrt(repetitions)),
            "reference": 1 - (1 + (slope - fit.theta) ** 2) / (1 + slope**2),
            "parameters": {"arm_n": n, "pilot_n": pilot_n, "pilot_theta": fit.theta,
                           "true_effect": effect, "true_slope": slope, "noise_sd": 1.0,
                           "unadjusted_effect_variance": raw_var, "adjusted_effect_variance": adjusted_var},
        })
    return output, rows


def _rate(check: str, metric: str, indicator: np.ndarray, reference: float | None, parameters: dict) -> dict:
    rate = float(np.mean(indicator))
    return {"check": check, "metric": metric, "estimate": rate, "reference": reference,
            "monte_carlo_se": sqrt(rate * (1 - rate) / len(indicator)), "parameters": parameters}


def _binary_p_values(xt, nt, xc, nc, alpha):
    pooled = (xt + xc) / (nt + nc)
    null_se = np.sqrt(pooled * (1 - pooled) * (1 / nt + 1 / nc))
    sparse = min(nt, nc) * np.minimum(pooled, 1 - pooled) < 5
    z = np.divide(xt / nt - xc / nc, null_se, out=np.zeros_like(pooled), where=null_se > 0)
    p_value = 2 * norm.sf(abs(z))
    for index in np.flatnonzero(sparse):
        p_value[index] = binary_from_counts(xt[index], nt, xc[index], nc,
                                            confidence_level=1 - alpha).p_value
    return p_value


def run_simulations(seed: int = 42, repetitions: int = 3000) -> list[dict]:
    validate_counts(repetitions, minimum=2)
    rng = np.random.default_rng(seed)
    alpha, baseline, alternative = 0.05, 0.10, 0.12
    plan = binary_sample_size(baseline, alternative, alpha=alpha)
    nc, nt = plan.control_n, plan.treated_n
    common = {"baseline": baseline, "control_n": nc, "treated_n": nt, "alpha": alpha}
    inspections = 10
    nc_steps = np.diff(np.linspace(0, nc, inspections + 1, dtype=int))
    nt_steps = np.diff(np.linspace(0, nt, inspections + 1, dtype=int))
    xc = rng.binomial(nc_steps, baseline, size=(repetitions, inspections)).cumsum(axis=1)
    xt = rng.binomial(nt_steps, baseline, size=(repetitions, inspections)).cumsum(axis=1)
    inspected = np.column_stack([
        _binary_p_values(xt[:, i], int(tn), xc[:, i], int(cn), alpha) < alpha
        for i, (tn, cn) in enumerate(zip(nt_steps.cumsum(), nc_steps.cumsum(), strict=True))
    ])
    schedule = common | {"null_treatment_rate": baseline,
                         "control_inspections": nc_steps.cumsum().tolist(),
                         "treated_inspections": nt_steps.cumsum().tolist()}
    rows = [
        _rate("fixed_horizon", "type_i_error", inspected[:, -1], alpha, schedule),
        _rate("naive_peeking", "type_i_error", inspected.any(axis=1), None, schedule),
    ]
    xc_alt, xt_alt = rng.binomial(nc, baseline, repetitions), rng.binomial(nt, alternative, repetitions)
    alt_params = common | {"alternative": alternative, "target_power": 0.8}
    rejection = _binary_p_values(xt_alt, nt, xc_alt, nc, alpha) < alpha
    rows.append(_rate("planned_alternative", "power", rejection, plan.achieved_power, alt_params))
    true_diff = alternative - baseline
    covered = np.empty(repetitions, dtype=bool)
    for i, (tx, cx) in enumerate(zip(xt_alt, xc_alt, strict=True)):
        lo, hi = binary_from_counts(tx, nt, cx, nc, confidence_level=1 - alpha).confidence_interval
        covered[i] = lo <= true_diff <= hi
    rows.append(_rate("newcombe_interval", "coverage", covered, 1 - alpha, alt_params | {"true_difference": true_diff}))

    family_size = 20
    families = rng.uniform(size=(repetitions, family_size))
    params = {"family_size": family_size, "alpha": alpha, "true_nulls": family_size}
    rows.append(_rate("uncorrected", "FWER", (families <= alpha).any(axis=1),
                      1 - (1 - alpha)**family_size, params))
    for method in MULTIPLICITY_METHODS:
        rejected = np.array([any(adjust_p_values(family, method, alpha=alpha).reject) for family in families])
        metric = "FDR" if method == "benjamini_hochberg" else "FWER"
        reference = alpha if metric == "FDR" else 1 - (1 - alpha / family_size)**family_size
        rows.append(_rate(method, metric, rejected, reference, params))
    return [row | {"seed": seed, "repetitions": repetitions} for row in rows]
