"""Configurable binary-outcome causal data simulator."""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from causallift.data.contract import DatasetSpec

FEATURES = ("engagement", "loyalty", "spend", "recency", "price_sensitivity", "activity")
TRUTH = ("mu0_true", "mu1_true", "tau_true", "propensity_true", "y0_true", "y1_true")
SCENARIOS = {
    "A": "Constant treatment effect",
    "B": "Nonlinear heterogeneous effect",
    "C": "Observed confounding",
    "D": "Imbalanced randomized assignment",
    "E": "Positive and negative effects",
    "F": "Hidden confounding and weak overlap",
}


def simulate(scenario: str, rows: int = 20_000, seed: int = 42):
    if scenario not in SCENARIOS or rows < 1:
        raise ValueError("Use scenario A–F and a positive row count")
    rng = np.random.default_rng(seed)
    covariates = rng.standard_normal((rows, len(FEATURES)))
    engagement, loyalty, spend, recency, price, activity = covariates.T
    baseline = -0.9 + 0.9 * engagement + 0.65 * loyalty + 0.4 * spend - 0.4 * recency
    mu0 = 0.2 + 0.45 * expit(baseline)
    nonlinear = 0.12 * np.sin(engagement) + 0.07 * np.tanh(price * activity)
    effects = {
        "A": 0.10,
        "B": nonlinear,
        "C": nonlinear,
        "D": nonlinear,
        "E": np.where(price >= 0, 0.16, -0.12),
        "F": nonlinear,
    }
    effect = effects[scenario]
    assignment = None if scenario in ("C", "F") else (0.15 if scenario == "D" else 0.5)
    propensity = np.full(rows, assignment) if assignment is not None else expit(
        -0.35 + 1.5 * engagement + 0.95 * loyalty
    )
    extra = {}
    assumptions = ("Independent units and independent Bernoulli potential-outcome draws given full covariates.",)
    if scenario == "F":
        latent = rng.standard_normal(rows)
        mu0 = 0.27 + 0.38 * expit(baseline + 1.5 * latent)
        propensity = 0.002 + 0.996 * expit(3 * engagement + 2.6 * latent)
        effect = nonlinear + 0.06 * (2 * norm.cdf(latent) - 1)
        extra = {"latent_confounder": latent, "tau_observed_true": nonlinear}
        assumptions += (
            "Exchangeability fails given observable covariates.",
            "tau_true conditions on the latent variable; tau_observed_true integrates it out.",
        )
    mu1 = mu0 + effect
    y0 = rng.binomial(1, mu0).astype("int8")
    y1 = rng.binomial(1, mu1).astype("int8")
    treatment = rng.binomial(1, propensity).astype("int8")
    observed = pd.DataFrame(covariates, columns=FEATURES).assign(
        unit_id=np.arange(rows), treatment=treatment, outcome=np.where(treatment, y1, y0),
    )
    truth = pd.DataFrame({
        "unit_id": observed["unit_id"], "mu0_true": mu0, "mu1_true": mu1,
        "tau_true": mu1 - mu0, "propensity_true": propensity, "y0_true": y0, "y1_true": y1,
        **extra,
    })
    spec = DatasetSpec(
        dataset_id=f"synthetic_{scenario}",
        source_type="synthetic_randomized" if assignment is not None else "synthetic_observational",
        treatment_column="treatment", outcome_column="outcome", feature_columns=FEATURES,
        unit_id="unit_id", assignment_probability=assignment,
        ground_truth_columns=(*TRUTH, *extra), assumptions=assumptions,
        provenance={"generator": "causallift.simulation.simulate", "scenario": SCENARIOS[scenario]},
    )
    spec.validate(observed, require_both_arms=False)
    return observed, truth, spec
