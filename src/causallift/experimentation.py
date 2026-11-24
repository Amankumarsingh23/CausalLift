"""Independent-arm analysis for binary and continuous outcomes."""

from dataclasses import dataclass
from math import hypot, isfinite, log, sqrt
from numbers import Integral

import numpy as np
from scipy.stats import fisher_exact, norm, t

Interval = tuple[float, float]
ASSUMPTIONS = (
    "Independent observations and treatment arms.",
    "Fixed-horizon, two-sided inference without multiplicity adjustment.",
    "A causal interpretation requires a design that identifies an assignment effect.",
)


@dataclass(frozen=True)
class ArmSummary:
    n: int
    mean: float
    events: int | None = None
    variance: float | None = None


@dataclass(frozen=True)
class ExperimentResult:
    treatment: ArmSummary
    control: ArmSummary
    absolute_lift: float
    relative_lift: float | None
    effect_se: float
    null_se: float | None
    statistic: float | None
    statistic_name: str | None
    p_value: float | None
    log10_p_value: float | None
    confidence_interval: Interval | None
    confidence_level: float
    significant: bool | None
    test_method: str
    interval_method: str
    dataset_id: str | None = None
    experiment_id: str | None = None
    degrees_of_freedom: float | None = None
    wald_interval: Interval | None = None
    bootstrap_interval: Interval | None = None
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ASSUMPTIONS

    @property
    def total_n(self) -> int:
        return self.treatment.n + self.control.n


def validated_rows(treatment, outcome, *, binary: bool):
    assignment, values = np.asarray(treatment), np.asarray(outcome)
    if (assignment.ndim != 1 or values.ndim != 1 or assignment.size != values.size
            or assignment.size == 0):
        raise ValueError("Treatment and outcome must be nonempty, equally sized 1D arrays")
    if assignment.dtype.kind not in "biuf" or not np.isin(assignment, [0, 1]).all():
        raise ValueError("Treatment assignment must be numeric or boolean 0/1")
    if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
        raise ValueError("Outcomes must be finite numeric values")
    if binary and not np.isin(values, [0, 1]).all():
        raise ValueError("Binary outcomes must be 0/1")
    return assignment.astype(bool), values


def _options(confidence_level, bootstrap_resamples):
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be strictly between zero and one")
    if (isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, Integral)
            or bootstrap_resamples == 1 or bootstrap_resamples < 0):
        raise ValueError("Bootstrap resamples must be zero or an integer of at least two")
    return (1 - confidence_level) / 2


def _wilson(rate: float, n: int, z: float) -> Interval:
    correction = z * z / n
    center = (rate + correction / 2) / (1 + correction)
    half_width = z * sqrt(rate * (1 - rate) / n + correction / (4 * n)) / (1 + correction)
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _percentile_interval(draws, tail) -> Interval:
    low, high = np.quantile(draws, [tail, 1 - tail])
    return float(low), float(high)


def binary_from_counts(
    treated_events: int, treated_n: int, control_events: int, control_n: int, *,
    confidence_level: float = 0.95, bootstrap_resamples: int = 0, seed: int = 42,
    dataset_id: str | None = None, experiment_id: str | None = None,
) -> ExperimentResult:
    """Analyze integer event counts. The effect is always treated minus control."""
    tail = _options(confidence_level, bootstrap_resamples)
    for events, n in ((treated_events, treated_n), (control_events, control_n)):
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (events, n)):
            raise ValueError("Arm sizes and event counts must be integers")
        if n <= 0 or not 0 <= events <= n:
            raise ValueError("Both arms must be nonempty with event counts between zero and arm size")
    xt, nt, xc, nc = map(int, (treated_events, treated_n, control_events, control_n))
    pt, pc = xt / nt, xc / nc
    diff = pt - pc
    se = sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)
    pooled = (xt + xc) / (nt + nc)
    null_se = sqrt(pooled * (1 - pooled) * (1 / nt + 1 / nc))
    min_expected = min(nt, nc) * min(pooled, 1 - pooled)
    warnings = []
    if pc == 0:
        warnings.append("Relative lift is undefined because control has no events.")
    if min_expected >= 5:
        method, statistic_name = "pooled_proportion_z", "z"
        statistic = diff / null_se
        log_p = float(log(2) + norm.logsf(abs(statistic)))
        p_value = float(np.exp(log_p))
        if p_value == 0:
            warnings.append("P-value underflowed to zero. Use log10_p_value.")
    else:
        method, statistic_name = "fisher_exact", "sample_odds_ratio"
        exact = fisher_exact([[xt, nt - xt], [xc, nc - xc]], alternative="two-sided")
        statistic = float(exact.statistic) if isfinite(exact.statistic) else None
        p_value = float(exact.pvalue)
        log_p = log(p_value) if p_value > 0 else None
        warnings.append("A null expected cell count is below five. Fisher and Newcombe decisions can differ.")
    if pooled in (0, 1):
        warnings.append("The outcome is constant across both arms. Fisher p=1 and the Wald interval collapses.")
    elif se == 0:
        warnings.append("Both arms have constant outcomes. The Wald interval collapses.")

    z = float(norm.isf(tail))
    t_low, t_high = _wilson(pt, nt, z)
    c_low, c_high = _wilson(pc, nc, z)
    newcombe = (diff - hypot(pt - t_low, c_high - pc), diff + hypot(t_high - pt, pc - c_low))
    bootstrap = None
    if bootstrap_resamples:
        rng = np.random.default_rng(seed)
        draws = rng.binomial(nt, pt, bootstrap_resamples) / nt - rng.binomial(nc, pc, bootstrap_resamples) / nc
        bootstrap = _percentile_interval(draws, tail)
        if bootstrap[0] == bootstrap[1]:
            warnings.append("The percentile bootstrap interval collapsed at the observed sample support.")
    return ExperimentResult(
        treatment=ArmSummary(nt, pt, xt), control=ArmSummary(nc, pc, xc),
        absolute_lift=diff, relative_lift=diff / pc if pc else None, effect_se=se,
        null_se=null_se, statistic=statistic, statistic_name=statistic_name,
        p_value=p_value, log10_p_value=log_p / log(10) if log_p is not None else None,
        confidence_interval=newcombe, confidence_level=confidence_level, significant=p_value < 2 * tail,
        test_method=method, interval_method="newcombe_wilson", dataset_id=dataset_id,
        experiment_id=experiment_id, wald_interval=(diff - z * se, diff + z * se),
        bootstrap_interval=bootstrap, warnings=tuple(warnings),
        assumptions=(*ASSUMPTIONS, "Binary outcomes follow the independent binomial sampling model."),
    )


def analyze_binary(treatment, outcome, **options) -> ExperimentResult:
    assigned, values = validated_rows(treatment, outcome, binary=True)
    treated, control = values[assigned], values[~assigned]
    return binary_from_counts(int(treated.sum()), len(treated), int(control.sum()), len(control), **options)


def analyze_continuous(
    treatment, outcome, *, confidence_level: float = 0.95, bootstrap_resamples: int = 0,
    seed: int = 42, dataset_id: str | None = None, experiment_id: str | None = None,
) -> ExperimentResult:
    tail = _options(confidence_level, bootstrap_resamples)
    assigned, values = validated_rows(treatment, outcome, binary=False)
    values = values.astype(float, copy=False)
    treated, control = values[assigned], values[~assigned]
    nt, nc = len(treated), len(control)
    if min(nt, nc) < 2:
        raise ValueError("Welch inference requires at least two observations per arm")
    mt, mc = float(treated.mean()), float(control.mean())
    vt, vc = float(treated.var(ddof=1)), float(control.var(ddof=1))
    diff = mt - mc
    scale_t, scale_c = vt / nt, vc / nc
    se = sqrt(scale_t + scale_c)
    if not all(map(isfinite, (mt, mc, vt, vc, diff, se))):
        raise ValueError("Continuous outcome summaries exceed floating-point range")
    relative = diff / mc if mc > 0 else None
    warnings = []
    if relative is None or not isfinite(relative):
        relative = None
        warnings.append("Relative lift requires a positive control mean and a finite ratio.")
    dof = statistic = p_value = log_p = interval = None
    if se:
        variance = scale_t + scale_c
        dof = 1 / ((scale_t / variance) ** 2 / (nt - 1) + (scale_c / variance) ** 2 / (nc - 1))
        statistic = diff / se
        log_p = float(log(2) + t.logsf(abs(statistic), dof))
        p_value = float(np.exp(log_p))
        half_width = float(t.isf(tail, dof)) * se
        interval = (diff - half_width, diff + half_width)
        if not isfinite(log_p):
            log_p = None
        if p_value == 0:
            warnings.append("P-value underflowed to zero." if log_p is None
                            else "P-value underflowed to zero. Use log10_p_value.")
    else:
        warnings.append("Both groups have zero estimated variance. Welch inference is undefined.")
    bootstrap = None
    if bootstrap_resamples:
        rng = np.random.default_rng(seed)
        draws = np.fromiter(
            (rng.choice(treated, size=nt).mean() - rng.choice(control, size=nc).mean()
             for _ in range(bootstrap_resamples)),
            dtype=float, count=bootstrap_resamples,
        )
        bootstrap = _percentile_interval(draws, tail)
        if bootstrap[0] == bootstrap[1]:
            warnings.append("The percentile bootstrap interval collapsed at the observed sample support.")
    return ExperimentResult(
        treatment=ArmSummary(nt, mt, variance=vt), control=ArmSummary(nc, mc, variance=vc),
        absolute_lift=diff, relative_lift=relative, effect_se=se, null_se=se if se else None,
        statistic=statistic, statistic_name="t" if se else None,
        p_value=p_value, log10_p_value=log_p / log(10) if log_p is not None else None,
        confidence_interval=interval, confidence_level=confidence_level,
        significant=p_value < 2 * tail if p_value is not None else None,
        test_method="welch_t" if se else "unavailable", interval_method="welch_t" if se else "unavailable",
        degrees_of_freedom=dof, bootstrap_interval=bootstrap, warnings=tuple(warnings),
        dataset_id=dataset_id, experiment_id=experiment_id,
        assumptions=(*ASSUMPTIONS, "Welch relies on normal populations or an adequate sampling approximation for arm means."),
    )
