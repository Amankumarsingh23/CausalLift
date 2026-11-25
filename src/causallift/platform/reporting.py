"""Aggregate executive exports from the published policy tables."""

LIMITATIONS = (
    "Criteo uses released validation assignment shares because original trial propensities "
    "are unavailable. Monetary values are hypothetical. The Criteo 1M validation cohort "
    "contains 58 control conversions. Fixed-action conditional intervals omit reoptimization, "
    "nuisance-fit uncertainty and unknown trial clustering. Scenario F is research-only."
)


def business_report(registry):
    lines = [
        "# CausalLift validation decision brief",
        "",
        "Population: fixed development validation cohorts, including ineligible units where applicable.",
        "Estimand: paired DR expected incremental profit relative to predictive targeting.",
        "Units: simulated or hypothetical currency. Intervals are conditional 95% Wald intervals.",
        "Provenance: Session 12 policy comparisons and Session 05 T-logistic outcome nuisance. "
        "Synthetic E uses Session 11 frozen allocations. Criteo uses Session 04 pooled LightGBM "
        "and Session 05 S-LightGBM.",
        "Economics: Synthetic E contribution 45 and cost 2 per treatment. Criteo uses "
        "synthetic unit contributions from 60 to 140 and cost 0.40 per treatment.",
        "",
    ]
    frame = registry.load("12/paired_policy_comparisons.csv")
    for dataset, economy, budget in (
        ("E", "uniform", 1600),
        ("Criteo 1M", "hypothetical", 16000),
    ):
        selected = frame.loc[
            frame.dataset.eq(dataset)
            & frame.economy.eq(economy)
            & frame.budget.eq(budget)
            & frame.method.eq("dr")
        ]
        if selected.empty:
            lines.extend([f"## {dataset}", "Published comparison unavailable.", ""])
            continue
        row = selected.iloc[0]
        lines.extend(
            [
                f"## {dataset}",
                f"Policy: {row.policy} versus {row.reference}. Budget ceiling: {budget:,}. "
                f"Proposed spend: {row.spend:,.2f} versus {row.reference_spend:,.2f}.",
                f"Estimated incremental profit: {row.incremental_profit:,.2f} "
                f"(95% CI {row.ci_profit_low:,.2f} to {row.ci_profit_high:,.2f}).",
                f"Population: {row.n:,}. Outcome nuisance: {row.outcome_nuisance}. "
                f"Identification: {row.identification}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            LIMITATIONS,
            "The Criteo profit-aware interval against treat-none includes zero.",
            "The final evaluation partition remains locked.",
            "",
        ]
    )
    return "\n".join(lines)
