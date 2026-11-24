"""Prespecified primary and guardrail decisions from experiment intervals."""

from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Literal

from causallift.experimentation import ExperimentResult
from causallift.planning import validate_alpha

Direction = Literal["higher", "lower"]
Health = Literal["pass", "fail", "not_assessable"]


@dataclass(frozen=True)
class Guardrail:
    metric: str
    direction: Direction
    harm_margin: float

    def __post_init__(self):
        _validate_metric(self.metric, self.direction, self.harm_margin)


@dataclass(frozen=True)
class ExperimentSpec:
    primary_metric: str
    direction: Direction
    alpha: float = 0.05
    minimum_practical_effect: float = 0.0
    guardrails: tuple[Guardrail, ...] = field(default_factory=tuple)

    def __post_init__(self):
        validate_alpha(self.alpha)
        _validate_metric(self.primary_metric, self.direction, self.minimum_practical_effect)
        metrics = (self.primary_metric, *(rule.metric for rule in self.guardrails))
        if len(set(metrics)) != len(metrics):
            raise ValueError("Primary and guardrail metrics must have unique names")


def _validate_metric(metric: str, direction: Direction, margin: float) -> None:
    if not metric or direction not in {"higher", "lower"} or not isfinite(margin) or margin < 0:
        raise ValueError("A metric needs a name, a higher/lower direction and a finite nonnegative margin")


def _check(
    result: ExperimentResult | None, direction: Direction, threshold: float, alpha: float, *, primary: bool,
) -> dict:
    status = "inconclusive"
    bounds = None
    if result is not None:
        if not isclose(result.confidence_level, 1 - alpha, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Metric confidence levels must match the prespecified decision alpha")
        if result.confidence_interval is not None:
            lo, hi = result.confidence_interval
            bounds = (lo, hi) if direction == "higher" else (-hi, -lo)
            if bounds[1] < threshold or (primary and bounds[1] == threshold):
                status = "fails"
            elif bounds[0] > threshold and (not primary or (result.p_value is not None and result.p_value < alpha)):
                status = "passes"
    return {"status": status, "benefit_interval": bounds, "direction": direction, "benefit_threshold": threshold}


def decide_experiment(
    spec: ExperimentSpec, results: dict[str, ExperimentResult], *, health: Health | None = None,
) -> dict:
    """Apply mandatory guardrail margins in the same units as the corresponding outcome."""
    if health not in {None, "pass", "fail", "not_assessable"}:
        raise ValueError("Unknown experiment health status")
    primary = _check(results.get(spec.primary_metric), spec.direction, spec.minimum_practical_effect,
                     spec.alpha, primary=True)
    guardrails = {
        rule.metric: _check(results.get(rule.metric), rule.direction, -rule.harm_margin, spec.alpha, primary=False)
        for rule in spec.guardrails
    }
    statuses = {item["status"] for item in (primary, *guardrails.values())}
    if health in {"fail", "not_assessable"}:
        decision, reason = "investigate", "Supplied experiment health did not pass."
    elif "fails" in statuses:
        decision, reason = "do_not_ship", "The primary target or a mandatory guardrail failed."
    elif statuses == {"passes"}:
        decision, reason = "ship", "The primary target and all mandatory guardrails passed."
    else:
        decision, reason = "inconclusive", "Evidence is insufficient for the primary target or guardrail safety."
    return {"decision": decision, "reason": reason, "health": health, "alpha": spec.alpha,
            "primary_metric": spec.primary_metric, "primary": primary, "guardrails": guardrails}
