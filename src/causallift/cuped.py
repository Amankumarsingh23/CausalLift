"""Pilot-fitted CUPED with independent evaluation and Welch inference."""

from dataclasses import dataclass

import numpy as np

from causallift.experimentation import ExperimentResult, analyze_continuous


def _paired_arrays(covariate, outcome) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(covariate), np.asarray(outcome)
    if (x.ndim != 1 or x.shape != y.shape or len(x) < 2
            or any(v.dtype.kind not in "biuf" or not np.isfinite(v).all() for v in (x, y))):
        raise ValueError("CUPED requires aligned, finite numeric covariate and outcome vectors")
    return x.astype(float, copy=False), y.astype(float, copy=False)


@dataclass(frozen=True)
class CupedFit:
    theta: float
    pilot_center: float
    pilot_correlation: float | None
    pilot_n: int
    warnings: tuple[str, ...]

    def adjust(self, covariate, outcome) -> np.ndarray:
        x, y = _paired_arrays(covariate, outcome)
        adjusted = y - self.theta * (x - self.pilot_center)
        if not np.isfinite(adjusted).all():
            raise ValueError("Adjusted outcomes exceed floating-point range")
        return adjusted


@dataclass(frozen=True)
class CupedResult:
    fit: CupedFit
    unadjusted: ExperimentResult
    adjusted: ExperimentResult
    variance_reduction: float | None


def fit_cuped(pilot_covariate, pilot_outcome) -> CupedFit:
    x, y = _paired_arrays(pilot_covariate, pilot_outcome)
    center = float(x.mean())
    if np.all(x == x[0]):
        return CupedFit(0.0, float(x[0]), None, len(x), ("Pilot covariate is constant. No adjustment applied.",))
    dx, dy = x - center, y - y.mean()
    xx = float(np.dot(dx, dx))
    yy = float(np.dot(dy, dy))
    xy = float(np.dot(dx, dy))
    if not np.isfinite([xx, yy, xy]).all() or xx == 0:
        raise ValueError("Pilot covariance exceeds supported floating-point precision")
    corr = float(np.clip((xy / np.sqrt(xx)) / np.sqrt(yy), -1, 1)) if yy else None
    warnings = ("Pilot outcome is constant. Correlation is undefined.",) if yy == 0 else ()
    return CupedFit(xy / xx, center, corr, len(x), warnings)


def analyze_cuped(
    treatment, outcome, pre_experiment_covariate, fit: CupedFit, *, confidence_level: float = 0.95,
) -> CupedResult:
    """Treat the independently fitted slope as fixed for evaluation inference."""
    adjusted_values = fit.adjust(pre_experiment_covariate, outcome)
    raw = analyze_continuous(treatment, outcome, confidence_level=confidence_level)
    adjusted = analyze_continuous(treatment, adjusted_values, confidence_level=confidence_level)
    reduction = 1 - (adjusted.effect_se / raw.effect_se) ** 2 if raw.effect_se else None
    return CupedResult(fit, raw, adjusted, reduction)
