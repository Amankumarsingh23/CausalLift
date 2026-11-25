"""Bounded platform request and response contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BinaryRequest(Inputs):
    treated_n: int = Field(ge=1, le=10_000_000, strict=True)
    control_n: int = Field(ge=1, le=10_000_000, strict=True)
    treated_events: int = Field(ge=0, strict=True)
    control_events: int = Field(ge=0, strict=True)
    confidence: float = Field(default=0.95, ge=0.80, le=0.999)

    @model_validator(mode="after")
    def events_within_arms(self):
        if self.treated_events > self.treated_n or self.control_events > self.control_n:
            raise ValueError("Event counts must be at most the corresponding arm size")
        return self


class PlanRequest(Inputs):
    baseline: float = Field(gt=0, lt=1)
    mde: float = Field(ge=0.00001, le=5)
    mde_kind: Literal["absolute", "relative"] = "absolute"
    alpha: float = Field(default=0.05, ge=0.001, le=0.2)
    power: float = Field(default=0.80, ge=0.5, le=0.99)
    ratio: float = Field(default=1, ge=0.1, le=10)


class PolicyRequest(Inputs):
    dataset: Literal["fixture_E", "fixture_F"] = "fixture_E"
    policy: Literal["none", "random", "predictive", "uplift", "profit"] = "profit"
    budget: float = Field(default=120, ge=0, le=100_000, multiple_of=0.01)
    capacity: int = Field(default=90, ge=0, le=240, strict=True)
    contribution: float = Field(default=45, ge=0, le=100_000)
    cost: float = Field(default=2, ge=0, le=10_000, multiple_of=0.01)


class BenchmarkResponse(BaseModel):
    artifact: str
    designation: str
    total: int
    rows: list[dict[str, Any]]


class PolicyResponse(BaseModel):
    dataset: str
    model: str
    split: str
    identification: str
    warning: str
    economics: PolicyRequest
    population: int
    eligible: int
    selected: int
    spend: float
    remaining_budget: float
    expected_conversions: float
    expected_contribution: float
    expected_profit: float
    expected_roi: float | None
    true_expected_synthetic_profit: float | None
    feasible: bool
    seed: int
    fixture_fingerprint: str
    allocation_fingerprint: str
    runtime_seconds: float
    units: dict[str, str]
