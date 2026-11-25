"""Versioned local-only API. File access is limited to the internal catalog."""

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from .artifact_registry import GROUPS, ArtifactRegistry
from .reporting import business_report
from .schemas import (
    BenchmarkResponse,
    BinaryRequest,
    PlanRequest,
    PolicyRequest,
    PolicyResponse,
)
from .services import analyze, benchmark, optimize, plan


def create_app(registry=None):
    registry = registry if registry is not None else ArtifactRegistry()
    app = FastAPI(title="CausalLift Decision API", version="1.0")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        issues = [
            {"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "invalid_request", "issues": issues}},
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "reserved_test": "locked"}

    @app.get("/api/v1/catalog")
    def catalog():
        return registry.catalog()

    @app.get("/api/v1/overview")
    def overview():
        try:
            return registry.load("01/dataset_summary.json")
        except FileNotFoundError as exc:
            raise HTTPException(404, detail={"code": "artifact_missing"}) from exc

    @app.get("/api/v1/{group}/benchmarks", response_model=BenchmarkResponse)
    def benchmarks(
        group: str,
        artifact: Annotated[str, Query(max_length=100)],
        dataset: Annotated[str | None, Query(max_length=100)] = None,
        outcome: Annotated[str | None, Query(max_length=40)] = None,
        model: Annotated[str | None, Query(max_length=80)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ):
        allowed = {
            f"{session:02d}/{name}"
            for session, names in GROUPS.get(group, {}).items()
            for name in names.split()
            if Path(name).suffix == ".csv"
        }
        if artifact not in allowed:
            raise HTTPException(404, detail={"code": "unknown_benchmark"})
        try:
            return benchmark(
                registry,
                artifact,
                dataset=dataset,
                outcome=outcome,
                model=model,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, detail={"code": "artifact_missing"}) from exc
        except KeyError as exc:
            raise HTTPException(404, detail={"code": "unknown_filter"}) from exc

    @app.get("/api/v1/reliability")
    def reliability():
        try:
            return registry.load("10/causal_reliability_report.json")
        except FileNotFoundError as exc:
            raise HTTPException(404, detail={"code": "artifact_missing"}) from exc

    @app.post("/api/v1/experiments/analyze")
    def binary(payload: BinaryRequest):
        return analyze(payload)

    @app.post("/api/v1/experiments/plan")
    def planning(payload: PlanRequest):
        try:
            return plan(payload)
        except ValueError as exc:
            raise HTTPException(
                422, detail={"code": "invalid_plan", "message": str(exc)}
            ) from exc

    @app.post("/api/v1/policies/optimize", response_model=PolicyResponse)
    def policy(payload: PolicyRequest):
        return optimize(payload)

    @app.get("/api/v1/reports/business", response_class=PlainTextResponse)
    def report():
        try:
            return PlainTextResponse(
                business_report(registry),
                media_type="text/markdown",
                headers={
                    "Content-Disposition": 'attachment; filename="causallift-decision-brief.md"'
                },
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, detail={"code": "artifact_missing"}) from exc

    return app


app = create_app()
