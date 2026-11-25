"""Shared application services for the API and dashboard."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from causallift.experimentation import binary_from_counts
from causallift.optimization import allocate, policy_totals
from causallift.planning import binary_sample_size

from .schemas import PolicyResponse

SEED = 42


def analyze(request):
    return asdict(
        binary_from_counts(
            request.treated_events,
            request.treated_n,
            request.control_events,
            request.control_n,
            confidence_level=request.confidence,
        )
    ) | {"units": "Rates, lift and confidence bounds are fractions."}


def plan(request):
    return asdict(
        binary_sample_size(
            request.baseline,
            **{f"{request.mde_kind}_effect": request.mde},
            alpha=request.alpha,
            target_power=request.power,
            ratio=request.ratio,
        )
    ) | {
        "units": "Participants per arm. Ratio is treated/control; power is a probability."
    }


def optimize(request):
    rng = np.random.default_rng(SEED)
    engagement, loyalty = rng.uniform(-1, 1, (2, 240))
    eligible = rng.random(240) < 0.9
    scores = 0.035 + 0.11 * engagement - 0.02 * loyalty
    predictive = 0.25 + 0.06 * engagement + 0.12 * loyalty
    fingerprint = hashlib.sha256(
        np.column_stack((engagement, loyalty, eligible, scores, predictive))
        .astype("<f8")
        .tobytes()
    ).hexdigest()
    ranking = {"random": np.zeros(240), "predictive": predictive, "uplift": scores}
    value = np.full(240, request.contribution)
    cost = np.full(240, request.cost)
    selected, diagnostics = allocate(
        value * scores - cost,
        cost,
        eligible,
        request.budget,
        capacity=request.capacity,
        policy={"none": "none", "profit": "uniform"}.get(request.policy, "ranking"),
        score=ranking.get(request.policy),
        seed=SEED,
    )
    projected = policy_totals(scores, value, cost, selected)
    identified = request.dataset == "fixture_E"
    truth_profit = None
    if identified:
        truth = 0.025 + 0.14 * engagement
        truth_profit = policy_totals(truth, value, cost, selected)["profit"]
    return PolicyResponse(
        dataset=request.dataset,
        model="frozen deterministic fixture scores v1",
        split="interactive fixture",
        identification="supported_under_assumptions"
        if identified
        else "not_identified",
        warning=(
            "Modeled projections under hypothetical economics."
            if identified
            else "Research-only illustration. Causal identification is unsupported."
        ),
        economics=request,
        population=240,
        eligible=int(eligible.sum()),
        selected=len(selected),
        spend=diagnostics["spend"],
        remaining_budget=round(request.budget - diagnostics["spend"], 2),
        expected_conversions=projected["conversions"],
        expected_contribution=projected["value"],
        expected_profit=projected["profit"],
        expected_roi=projected["roi"],
        true_expected_synthetic_profit=truth_profit,
        feasible=diagnostics["feasible"],
        seed=SEED,
        fixture_fingerprint=fingerprint,
        allocation_fingerprint=hashlib.sha256(
            np.sort(selected).astype("<i8").tobytes()
        ).hexdigest(),
        runtime_seconds=diagnostics["runtime_seconds"],
        units={
            "money": "hypothetical currency units",
            "roi": "fraction",
            "capacity": "customers",
        },
    )


def benchmark(registry, key, *, dataset=None, outcome=None, model=None, limit=100):
    frame = registry.load(key)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("This result is not a benchmark table")
    for value, columns in (
        (dataset, ("dataset", "dataset_id", "scenario")),
        (outcome, ("outcome", "outcome_name", "endpoint")),
        (model, ("policy", "model", "estimator", "method", "learner")),
    ):
        if value is None:
            continue
        column = next((name for name in columns if name in frame), None)
        if column is None or not frame[column].eq(value).any():
            raise KeyError("Filter value is unavailable for this benchmark")
        frame = frame.loc[frame[column].eq(value)]
    return {
        "artifact": key,
        "designation": "published development benchmark",
        "total": len(frame),
        "rows": json.loads(
            frame.head(limit).to_json(orient="records", double_precision=15)
        ),
    }
