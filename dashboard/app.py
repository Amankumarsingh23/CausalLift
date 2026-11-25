"""Streamlit application for CausalLift's development decision workflow."""

import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from causallift.platform.artifact_registry import GROUPS, ArtifactRegistry
from causallift.platform.reporting import LIMITATIONS, business_report
from causallift.platform.schemas import BinaryRequest, PlanRequest, PolicyRequest
from causallift.platform.services import analyze, optimize, plan

PAGES = (
    "Overview",
    "Experimentation lab",
    "Uplift comparison",
    "Observational inference",
    "Causal reliability",
    "Business policy lab",
    "Reports",
)
API = os.environ.get("CAUSALLIFT_API")
registry = ArtifactRegistry()

st.set_page_config(
    page_title="CausalLift Decision Studio", page_icon="📈", layout="wide"
)
with st.sidebar:
    st.title("CausalLift")
    st.caption("Decision studio")
    page = st.radio("Workspace", PAGES)
    st.divider()
    st.caption("Development validation · Final test locked")


def load(key):
    try:
        suffix = Path(key).suffix
        return registry.path(key) if suffix == ".png" else registry.load(key)
    except FileNotFoundError:
        st.warning(f"Published artifact unavailable: {key}")
        return None


def show(key, *, title=None, drop_truth=False):
    st.subheader(
        title or f"Session {key[:2]} · {Path(key).stem.replace('_', ' ').title()}"
    )
    result = load(key)
    if isinstance(result, pd.DataFrame):
        display = (
            result.drop(
                columns=[
                    c for c in result if c.startswith("true_") or c == "profit_error"
                ]
            )
            if drop_truth
            else result
        )
        st.dataframe(display, width="stretch", hide_index=True)
    elif isinstance(result, Path):
        st.image(str(result))
    elif result is not None:
        st.json(result, expanded=False)
    return result


def execute(route, service, schema, values):
    try:
        payload = schema(**values)
        if API:
            result = httpx.post(
                f"{API}/api/v1/{route}", json=payload.model_dump(), timeout=8
            )
            if result.is_error:
                st.error(result.json()["detail"])
                return None
            return result.json()
        result = service(payload)
        return result.model_dump() if hasattr(result, "model_dump") else result
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
    except httpx.HTTPError:
        st.error("The configured decision API is unavailable.")
    return None


def metrics(values):
    for column, (name, value) in zip(st.columns(len(values)), values, strict=True):
        column.metric(name, value)


def overview():
    st.caption("DEVELOPMENT VALIDATION")
    st.title("Evidence for treatment decisions")
    st.write(
        "Plan experiments, compare causal models, inspect assumptions, and test a treatment budget."
    )
    catalog = registry.catalog()
    models = catalog["models"]
    available = sum(
        row["status"] == "published_aggregate" for row in catalog["artifacts"]
    )
    metrics(
        [
            ("Published artifacts", available),
            ("Model configurations", len(models)),
            ("Capability", catalog["mode"]),
            ("Evaluation partition", "Validation"),
        ]
    )
    st.info(
        "Published development results, synthetic expected truth, Criteo released-sample estimates, and interactive fixtures have different interpretations. Criteo economics are hypothetical."
    )
    show("02/synthetic_experiment_results.csv", title="Experiment evidence")
    ranking = load("05/uplift_ranking_metrics.csv")
    if ranking is not None:
        st.subheader("Synthetic uplift evidence")
        st.dataframe(
            ranking.loc[ranking.dataset.str.contains("synthetic", case=False)].head(10),
            hide_index=True,
        )
    paired = load("12/paired_policy_comparisons.csv")
    if paired is not None:
        st.subheader("Profit-aware versus predictive targeting")
        st.dataframe(
            paired.query(
                "method == 'dr' and ((dataset == 'E' and economy == 'uniform' and budget == 1600) or (dataset == 'Criteo 1M' and budget == 16000))"
            )[
                [
                    "dataset",
                    "spend",
                    "reference_spend",
                    "incremental_profit",
                    "ci_profit_low",
                    "ci_profit_high",
                ]
            ],
            hide_index=True,
        )
    st.warning(LIMITATIONS)
    with st.expander("Data, benchmark and model inventory"):
        show("01/dataset_summary.json")
        show("01/split_summary.csv")
        st.dataframe(catalog["artifacts"], hide_index=True)
        st.dataframe(models, hide_index=True)


def experimentation():
    st.title("Experimentation lab")
    st.caption("Interactive inputs · Independent binary arms")
    analysis, planning, evidence = st.tabs(
        ["Analyze an A/B test", "Plan sample size", "Published evidence"]
    )
    with analysis:
        with st.form("binary"):
            left, right = st.columns(2)
            nt = left.number_input("Treatment sample size", 1, 10_000_000, 1000)
            xt = left.number_input("Treatment conversions", 0, 10_000_000, 120)
            nc = right.number_input("Control sample size", 1, 10_000_000, 1000)
            xc = right.number_input("Control conversions", 0, 10_000_000, 100)
            confidence = st.slider("Confidence level", 0.80, 0.99, 0.95)
            submitted = st.form_submit_button("Analyze experiment", type="primary")
        if submitted:
            out = execute(
                "experiments/analyze",
                analyze,
                BinaryRequest,
                dict(
                    treated_n=nt,
                    control_n=nc,
                    treated_events=xt,
                    control_events=xc,
                    confidence=confidence,
                ),
            )
            if out:
                metrics(
                    [
                        ("Treatment rate", f"{out['treatment']['mean']:.2%}"),
                        ("Control rate", f"{out['control']['mean']:.2%}"),
                        ("Absolute lift", f"{100 * out['absolute_lift']:.2f} pp"),
                        ("P-value", f"{out['p_value']:.4g}"),
                    ]
                )
                relative = out["relative_lift"]
                lo, hi = out["confidence_interval"]
                relative_label = "undefined" if relative is None else f"{relative:.2%}"
                st.write(
                    f"Relative lift {relative_label} · {confidence:.0%} CI [{100 * lo:.2f}, {100 * hi:.2f}] pp · {out['test_method']} · {out['interval_method']}"
                )
                for warning in out["warnings"]:
                    st.warning(warning)
    with planning:
        with st.form("planning"):
            baseline = st.number_input(
                "Baseline probability", 0.00001, 0.99999, 0.10, format="%.5f"
            )
            kind = st.selectbox(
                "Minimum detectable effect type", ["absolute", "relative"]
            )
            mde = st.number_input(
                "Minimum detectable effect (fraction)",
                0.00001,
                5.0,
                0.02,
                format="%.5f",
            )
            alpha = st.number_input(
                "Significance level", 0.001, 0.2, 0.05, format="%.3f"
            )
            power = st.slider("Desired power", 0.50, 0.99, 0.80)
            ratio = st.number_input(
                "Treatment / control allocation ratio", 0.1, 10.0, 1.0
            )
            submitted = st.form_submit_button("Calculate sample size", type="primary")
        if submitted:
            out = execute(
                "experiments/plan",
                plan,
                PlanRequest,
                dict(
                    baseline=baseline,
                    mde=mde,
                    mde_kind=kind,
                    alpha=alpha,
                    power=power,
                    ratio=ratio,
                ),
            )
            if out:
                metrics(
                    [
                        ("Treatment required", out["treated_n"]),
                        ("Control required", out["control_n"]),
                        ("Achieved power", f"{out['achieved_power']:.2%}"),
                    ]
                )
                for warning in out.get("warnings", []):
                    st.warning(warning)
    with evidence:
        st.write(
            "CUPED uses pre-treatment covariates to improve precision. Peeking and repeated testing can inflate false positives."
        )
        for session, names in GROUPS["experiments"].items():
            for name in names.split():
                with st.expander(name.replace("_", " ").split(".")[0].title()):
                    show(f"{session:02d}/{name}")


def uplift():
    st.title("Uplift model comparison")
    st.warning(
        "Criteo rankings use released-sample assignment approximations and rare conversion outcomes. Root PEHE is available for synthetic scenarios."
    )
    session = st.selectbox("Benchmark session", [5, 6, 7])
    key = f"{session:02d}/uplift_ranking_metrics.csv"
    ranking = load(key)
    if ranking is not None:
        columns = st.columns(3)
        filters = {}
        for column, field, label in zip(
            columns,
            ["dataset", "outcome", "policy"],
            ["Dataset", "Outcome", "Model"],
            strict=True,
        ):
            if field in ranking:
                values = sorted(ranking[field].dropna().unique())
                choice = column.selectbox(label, ["All", *values])
                if choice != "All":
                    filters[field] = choice
                    ranking = ranking.loc[ranking[field].eq(choice)]
        st.dataframe(ranking, hide_index=True, width="stretch")
        if session == 5:
            curve = load("05/uplift_curve_coordinates.csv")
            if curve is not None:
                for field, value in filters.items():
                    if field in curve:
                        curve = curve.loc[curve[field].eq(value)]
                if not curve.empty:
                    cohort = st.selectbox(
                        "Curve cohort",
                        curve[["dataset", "outcome"]]
                        .drop_duplicates()
                        .agg(" / ".join, axis=1),
                    )
                    curve = curve.loc[
                        (curve.dataset + " / " + curve.outcome).eq(cohort)
                    ]
                    st.plotly_chart(
                        px.line(
                            curve,
                            x="fraction",
                            y="qini",
                            color="policy",
                            title="Stored Qini curve coordinates",
                        )
                    )
        if session != 5:
            show("07/causal_ml_comparison.png", title="Published causal ML comparison")
    with st.expander("Prediction, calibration, accuracy and runtime"):
        for key in (
            "04/predictive_model_results.csv",
            "04/synthetic_predictive_results.csv",
            f"{session:02d}/synthetic_effect_accuracy.csv",
            f"{session:02d}/uplift_calibration.csv",
            {
                5: "05/causal_estimator_results.csv",
                6: "06/meta_learner_comparison.csv",
                7: "07/final_causal_ml_comparison.csv",
            }[session],
        ):
            show(key)


def observational():
    st.title("Observational causal inference")
    st.info(
        "ATE targets the cohort. ATT targets treated units. Matched ATT targets the retained matched population."
    )
    for session, names in GROUPS["observational"].items():
        for name in names.split():
            with st.expander(name.replace("_", " ").split(".")[0].title()):
                show(f"{session:02d}/{name}")
    st.caption(
        "NSW/CPS is an external comparison. Session 09 preserves the original matching failure. Session 10 records the outcome-independent tie correction separately."
    )


def reliability():
    st.title("Causal reliability")
    st.warning(
        "Scenario F remains research-only. Session 07 DML coverage was 0/30 in the reported simulation."
    )
    result = load("10/causal_reliability_report.json")
    if result:
        dataset = st.selectbox("Scenario", list(result["datasets"]))
        record = result["datasets"][dataset]
        st.metric("Identification status", record["status"])
        st.json(record, expanded=1)
    for name in GROUPS["reliability"][10].split()[1:]:
        with st.expander(name.replace("_", " ").split(".")[0].title()):
            show(f"10/{name}")
    with st.expander("DML inference diagnostics"):
        show("07/dml_diagnostics.json")


def policy_lab():
    st.title("Business policy lab")
    st.caption("INTERACTIVE FIXTURE · 240 SYNTHETIC CUSTOMERS")
    st.write(
        "Change the budget and unit economics to see which treatments the allocation engine proposes."
    )
    left, right = st.columns([1, 2])
    with left:
        with st.form("policy"):
            dataset = st.selectbox(
                "Population",
                ["fixture_E", "fixture_F"],
                format_func=lambda name: {
                    "fixture_E": "Scenario E fixture",
                    "fixture_F": "Scenario F research fixture",
                }[name],
            )
            policy = st.selectbox(
                "Allocation policy",
                ["profit", "uplift", "predictive", "random", "none"],
                format_func=lambda name: {
                    "profit": "Profit-aware",
                    "uplift": "Estimated uplift",
                    "predictive": "Predictive ranking",
                    "random": "Random",
                    "none": "Treat none",
                }[name],
            )
            budget = st.number_input(
                "Budget (currency units)",
                0.0,
                100000.0,
                120.0,
                step=10.0,
                format="%.2f",
            )
            capacity = st.number_input("Maximum customers", 0, 240, 90)
            value = st.number_input("Contribution per conversion", 0.0, 100000.0, 45.0)
            cost = st.number_input(
                "Fixed cost per treatment", 0.0, 10000.0, 2.0, format="%.2f"
            )
            st.form_submit_button("Run allocation", type="primary")
    with right:
        out = execute(
            "policies/optimize",
            optimize,
            PolicyRequest,
            dict(
                dataset=dataset,
                policy=policy,
                budget=budget,
                capacity=capacity,
                contribution=value,
                cost=cost,
            ),
        )
        if out:
            metrics(
                [
                    ("Customers selected", out["selected"]),
                    ("Proposed spend", f"{out['spend']:,.2f}"),
                    ("Modeled profit", f"{out['expected_profit']:,.2f}"),
                ]
            )
            roi = out["expected_roi"]
            metrics(
                [
                    ("Unspent budget", f"{out['remaining_budget']:,.2f}"),
                    ("Expected conversions", f"{out['expected_conversions']:,.2f}"),
                    ("Expected ROI", "Undefined" if roi is None else f"{roi:.1%}"),
                ]
            )
            st.write(
                f"Expected contribution {out['expected_contribution']:,.2f} · Eligible {out['eligible']}/{out['population']} · Feasible {out['feasible']}"
            )
            if out["identification"] == "not_identified":
                st.warning(out["warning"])
            else:
                st.info(f"Supported under fixture assumptions. {out['warning']}")
            if out["true_expected_synthetic_profit"] is not None:
                st.caption(
                    f"Synthetic expected truth, scored after allocation: {out['true_expected_synthetic_profit']:,.2f}"
                )
    chart = load("12/budget_profit_frontier.csv")
    if chart is not None:
        chart = chart.query("dataset == 'E' and economy == 'uniform'")
        st.plotly_chart(
            px.line(
                chart,
                x="budget",
                y="true_profit",
                color="policy",
                markers=True,
                title="Published Scenario E validation · synthetic expected profit",
                labels={
                    "budget": "Budget ceiling",
                    "true_profit": "Synthetic currency units",
                },
            )
        )


def reports():
    st.title("Policy evaluation and reports")
    st.warning(LIMITATIONS)
    st.write(
        "Criteo profit-aware versus treat-none includes zero in its interval. The paired comparison to predictive targeting uses different proposed spending."
    )
    key = st.selectbox(
        "Validation result",
        GROUPS["evaluation"][12].split(),
        format_func=lambda name: Path(name).stem.replace("_", " ").title(),
    )
    table = show(f"12/{key}", title="Published aggregate results", drop_truth=True)
    if isinstance(table, pd.DataFrame):
        columns = [c for c in table if c.startswith("true_") or c == "profit_error"]
        st.download_button(
            "Download aggregate CSV",
            table.drop(columns=columns).to_csv(index=False),
            file_name=key,
            mime="text/csv",
        )
        if key == "cost_sensitivity.csv":
            sensitivity = table.loc[table.value.eq(45)]
            st.plotly_chart(
                px.line(
                    sensitivity,
                    x="cost",
                    y="true_profit",
                    color="policy",
                    markers=True,
                    title="Synthetic expected profit at contribution 45",
                )
            )
    try:
        brief = business_report(registry)
    except FileNotFoundError:
        st.warning(
            "The decision brief requires the published paired policy comparisons."
        )
    else:
        st.download_button(
            "Download executive brief",
            brief,
            file_name="causallift-decision-brief.md",
            mime="text/markdown",
        )
        with st.expander("Preview executive brief"):
            st.markdown(brief)


dict(
    zip(
        PAGES,
        (
            overview,
            experimentation,
            uplift,
            observational,
            reliability,
            policy_lab,
            reports,
        ),
        strict=True,
    )
)[page]()
