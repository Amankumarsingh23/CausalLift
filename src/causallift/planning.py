"""Two-sided normal-approximation planning for independent arms."""

from dataclasses import dataclass
from math import asin, ceil, isfinite, sqrt
from numbers import Integral

from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class SampleSize:
    control_n: int
    treated_n: int
    achieved_power: float
    warnings: tuple[str, ...] = ()

    @property
    def total_n(self) -> int:
        return self.control_n + self.treated_n


@dataclass(frozen=True)
class DetectableEffect:
    alternative: float
    absolute_effect: float
    relative_effect: float
    achieved_power: float
    warnings: tuple[str, ...]


def validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie strictly between zero and one")


def validate_counts(*counts: int, minimum: int) -> None:
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < minimum for n in counts):
        raise ValueError(f"Arm counts must be integers of at least {minimum}")


def _effect(baseline: float, alternative: float) -> float:
    if not (0 < baseline < 1 and 0 <= alternative <= 1):
        raise ValueError("Baseline must be in (0, 1) and alternative in [0, 1]")
    # Using the complementary probability near one avoids cancellation in the transform.
    if baseline > 0.5 and alternative > 0.5:
        return 2 * (asin(sqrt(1 - baseline)) - asin(sqrt(1 - alternative)))
    return 2 * (asin(sqrt(alternative)) - asin(sqrt(baseline)))


def _power(shift: float, critical: float) -> float:
    return float(norm.sf(critical - shift) + norm.sf(critical + shift))


def _critical_and_shift(alpha: float, target_power: float) -> tuple[float, float]:
    validate_alpha(alpha)
    if not alpha < target_power < 1:
        raise ValueError("Target power must lie strictly between alpha and one")
    critical = float(norm.isf(alpha / 2))
    shift = brentq(lambda value: _power(value, critical) - target_power,
                  0, critical + float(norm.isf((1 - target_power) / 2)), xtol=1e-14)
    return critical, shift


def _effective_n(control_n: int, treated_n: int) -> float:
    return 1 / (1 / control_n + 1 / treated_n)


def _rare_warning(baseline: float, alternative: float, nc: int, nt: int) -> tuple[str, ...]:
    pooled = (nc * baseline + nt * alternative) / (nc + nt)
    if min(nc * min(baseline, 1 - baseline), nt * min(alternative, 1 - alternative),
           min(nc, nt) * min(pooled, 1 - pooled)) < 5:
        return ("An expected event or non-event count is below five. Normal planning may be unreliable.",)
    return ()


def _sample_size(effect: float, ratio: float, alpha: float, target_power: float) -> SampleSize:
    if not isfinite(effect) or effect == 0:
        raise ValueError("Planning requires a finite, nonzero effect")
    if not isfinite(ratio) or ratio <= 0:
        raise ValueError("Treatment/control ratio must be finite and positive")
    critical, shift = _critical_and_shift(alpha, target_power)
    scaled_shift = shift / abs(effect)
    effective = scaled_shift * scaled_shift
    control = effective * (1 + 1 / ratio)
    treated = effective * (1 + ratio)
    if not all(map(isfinite, (control, treated))) or max(control, treated) >= 2**53:
        raise ValueError("The required arm size exceeds supported integer precision")
    nc, nt = max(2, ceil(control)), max(2, ceil(treated))
    power = _power(abs(effect) * sqrt(_effective_n(nc, nt)), critical)
    if power < target_power:
        nc, nt = nc + 1, nt + 1
        power = _power(abs(effect) * sqrt(_effective_n(nc, nt)), critical)
    return SampleSize(nc, nt, power)


def binary_sample_size(
    baseline: float, alternative: float | None = None, *, absolute_effect: float | None = None,
    relative_effect: float | None = None, ratio: float = 1.0,
    alpha: float = 0.05, target_power: float = 0.80,
) -> SampleSize:
    if sum(x is not None for x in (alternative, absolute_effect, relative_effect)) != 1:
        raise ValueError("Supply exactly one alternative probability, absolute effect, or relative effect")
    if absolute_effect is not None:
        alternative = baseline + absolute_effect
    elif relative_effect is not None:
        alternative = baseline * (1 + relative_effect)
    plan = _sample_size(_effect(baseline, alternative), ratio, alpha, target_power)
    return SampleSize(plan.control_n, plan.treated_n, plan.achieved_power,
                      _rare_warning(baseline, alternative, plan.control_n, plan.treated_n))


def binary_power(
    baseline: float, alternative: float, control_n: int, treated_n: int, *, alpha: float = 0.05,
) -> float:
    """Approximate power at fixed arm sizes. See binary_power_warnings for sparse cells."""
    return continuous_power(_effect(baseline, alternative), control_n, treated_n, alpha=alpha)


def binary_power_warnings(baseline: float, alternative: float, control_n: int, treated_n: int) -> tuple[str, ...]:
    _effect(baseline, alternative)
    validate_counts(control_n, treated_n, minimum=1)
    return _rare_warning(baseline, alternative, control_n, treated_n)


def binary_mde(
    baseline: float, control_n: int, treated_n: int, *, direction: str = "increase",
    alpha: float = 0.05, target_power: float = 0.80,
) -> DetectableEffect:
    validate_counts(control_n, treated_n, minimum=1)
    if direction not in {"increase", "decrease"}:
        raise ValueError("Direction must be increase or decrease")
    bound = 1.0 if direction == "increase" else 0.0
    max_effect = abs(_effect(baseline, bound))
    critical, shift = _critical_and_shift(alpha, target_power)
    scale = sqrt(_effective_n(control_n, treated_n))
    required = shift / scale
    if required > max_effect:
        raise ValueError("Target power is infeasible within the requested probability bounds")
    alternative = brentq(lambda p: abs(_effect(baseline, p)) - required,
                         min(baseline, bound), max(baseline, bound), xtol=5e-324)
    diff = alternative - baseline
    relative = diff / baseline
    if not isfinite(relative):
        raise ValueError("The relative detectable effect exceeds floating-point range")
    return DetectableEffect(alternative, diff, relative,
                            _power(abs(_effect(baseline, alternative)) * scale, critical),
                            _rare_warning(baseline, alternative, control_n, treated_n))


def continuous_sample_size(
    standardized_difference: float, *, ratio: float = 1.0,
    alpha: float = 0.05, target_power: float = 0.80,
) -> SampleSize:
    """Plan for a mean difference divided by a common population standard deviation."""
    return _sample_size(standardized_difference, ratio, alpha, target_power)


def continuous_power(
    standardized_difference: float, control_n: int, treated_n: int, *, alpha: float = 0.05,
) -> float:
    validate_counts(control_n, treated_n, minimum=1)
    validate_alpha(alpha)
    if not isfinite(standardized_difference):
        raise ValueError("Standardized difference must be finite")
    return _power(abs(standardized_difference) * sqrt(_effective_n(control_n, treated_n)),
                  float(norm.isf(alpha / 2)))
