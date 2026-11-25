"""Run the offline decision platform verification and write aggregate evidence."""

import argparse
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd
from fastapi.testclient import TestClient

from .platform.api import create_app
from .platform.artifact_registry import FROZEN_HASHES, PATHS, ArtifactRegistry
from .platform.schemas import PolicyRequest, PolicyResponse


def run(root, output, tracking=None):
    source = ArtifactRegistry(root)
    before = source.protocol_checksum()
    if before not in FROZEN_HASHES:
        raise ValueError("Frozen final protocol differs from Session 12")
    output.mkdir(parents=True, exist_ok=True)
    measures = []
    smoke = {}

    def timed(name, call, repeat=1):
        for _ in range(repeat):
            start = perf_counter()
            result = call()
            measures.append(
                {"operation": name, "milliseconds": (perf_counter() - start) * 1000}
            )
        return result

    with TemporaryDirectory(prefix="causallift-demo-") as isolated:
        isolated = Path(isolated)
        for key in PATHS:
            path = source.path(key)
            target = isolated / path.relative_to(source.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        registry = ArtifactRegistry(isolated)
        with TestClient(create_app(registry)) as client:

            def request(name, path, payload=None, status=200):
                if payload is None:
                    response = timed(name, lambda: client.get(path), repeat=3)
                else:
                    response = timed(
                        name, lambda: client.post(path, json=payload), repeat=3
                    )
                if response.status_code != status:
                    raise RuntimeError(
                        f"{name} returned {response.status_code}, expected {status}"
                    )
                smoke[name] = status
                return response

            request("health", "/health")
            catalog = request("catalog", "/api/v1/catalog").json()
            if catalog["mode"] != "artifact-only":
                raise RuntimeError(
                    "Offline verification unexpectedly detected local assets"
                )
            ab = request(
                "binary",
                "/api/v1/experiments/analyze",
                {
                    "treated_n": 1000,
                    "treated_events": 120,
                    "control_n": 1000,
                    "control_events": 100,
                },
            ).json()
            plan = request(
                "planning", "/api/v1/experiments/plan", {"baseline": 0.1, "mde": 0.02}
            ).json()
            timed(
                "artifact_cold_load",
                lambda: registry.load("12/criteo_policy_evaluation.csv"),
            )
            request(
                "benchmark",
                "/api/v1/uplift/benchmarks?artifact=05/uplift_ranking_metrics.csv&limit=10",
            )
            request("reliability", "/api/v1/reliability")
            route = "/api/v1/policies/optimize"
            low = request(
                "policy_budget_60", route, PolicyRequest(budget=60).model_dump()
            ).json()
            high = request(
                "policy_budget_120", route, PolicyRequest().model_dump()
            ).json()
            if low["selected"] >= high["selected"] or not high["feasible"]:
                raise RuntimeError("Fixture allocation failed the budget comparison")
            report = request("executive_export", "/api/v1/reports/business")
            if "attachment" not in report.headers["Content-Disposition"]:
                raise RuntimeError("Executive report is missing its download header")
            registry.path("09/weight_diagnostics.csv").unlink()
            request(
                "missing_artifact",
                "/api/v1/observational/benchmarks?artifact=09/weight_diagnostics.csv",
                status=404,
            )
    tracked = None
    if tracking:
        from .platform.tracking import log_policy

        tracked = log_policy(PolicyResponse.model_validate(high), tracking)
    after = source.protocol_checksum()
    if before != after:
        raise RuntimeError("Frozen final evaluation protocol changed")
    verification = {
        "mode": catalog["mode"],
        "asset_count": len(catalog["artifacts"]),
        "protocol_before": before,
        "protocol_after": after,
        "reserved_test_accessed": False,
        "environment": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "fastapi",
                "uvicorn",
                "streamlit",
                "plotly",
                "pydantic",
                "httpx",
                "numpy",
                "pandas",
            )
        },
        "benchmark_method": "Perf counter, in-process TestClient or cold CSV parse, three calls per route",
        "mlflow_run_id": tracked,
    }
    for name, value in {
        "platform_verification": verification,
        "api_smoke_results": smoke,
        "demo_run_manifest": {
            "analysis": ab,
            "planning": plan,
            "budget_60": low,
            "budget_120": high,
        },
    }.items():
        (output / f"{name}.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    pd.DataFrame(measures).groupby("operation").milliseconds.agg(
        ["count", "median", "max"]
    ).to_csv(output / "response_time_summary.csv", float_format="%.3f")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/session_13"))
    parser.add_argument("--tracking-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, args.tracking_dir), indent=2))


if __name__ == "__main__":
    main()
