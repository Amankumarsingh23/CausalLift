import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("streamlit")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from causallift.experimentation import binary_from_counts  # noqa: E402
from causallift.planning import binary_sample_size  # noqa: E402
from causallift.platform.api import create_app  # noqa: E402
from causallift.platform.artifact_registry import FROZEN_HASHES, PATHS, ArtifactRegistry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def aggregates(tmp_path, monkeypatch):
    source = ArtifactRegistry(ROOT)
    for key in PATHS:
        origin = source.path(key)
        target = tmp_path / origin.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    monkeypatch.setenv("CAUSALLIFT_ROOT", str(tmp_path))
    monkeypatch.delenv("CAUSALLIFT_API", raising=False)
    return ArtifactRegistry(tmp_path)


@pytest.fixture
def client(aggregates):
    with TestClient(create_app(aggregates)) as client:
        yield client


def test_registry_missing_and_safe_paths(aggregates, client):
    assert client.get("/health").json()["status"] == "ok"
    catalog = client.get("/api/v1/catalog").json()
    assert catalog["mode"] == "artifact-only"
    assert catalog["reserved_test"] == "locked"
    assert not catalog["local_row_data"]
    assert catalog["models"]
    assert {row["status"] for row in catalog["models"]} == {"missing"}
    with pytest.raises(KeyError):
        aggregates.path("../../.env")
    with pytest.raises(KeyError):
        aggregates.path("12/final_evaluation_protocol.json")
    aggregates.path("09/weight_diagnostics.csv").unlink()
    response = client.get(
        "/api/v1/observational/benchmarks?artifact=09/weight_diagnostics.csv"
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "artifact_missing"
    assert (
        client.get(
            "/api/v1/observational/benchmarks?artifact=../../data/test.parquet"
        ).status_code
        == 404
    )
    assert ArtifactRegistry(ROOT).protocol_checksum() in FROZEN_HASHES
    assert not (aggregates.root / "data").exists()


def test_api_calculations_and_filter_validation(client):
    route = "/api/v1/experiments/analyze"
    response = client.post(
        route,
        json={
            "treated_n": 1000,
            "treated_events": 120,
            "control_n": 1000,
            "control_events": 100,
        },
    ).json()
    expected = binary_from_counts(120, 1000, 100, 1000)
    assert response["absolute_lift"] == expected.absolute_lift
    assert response["confidence_interval"] == list(expected.confidence_interval)
    assert response["p_value"] == expected.p_value
    assert (
        client.post(route, json={"treated_n": True}).json()["detail"]["code"]
        == "invalid_request"
    )
    route = "/api/v1/experiments/plan"
    response = client.post(route, json={"baseline": 0.1, "mde": 0.02}).json()
    expected = binary_sample_size(0.1, absolute_effect=0.02)
    assert response["treated_n"] == expected.treated_n
    assert response["achieved_power"] == expected.achieved_power
    assert client.post(route, json={"baseline": 0.9, "mde": 0.2}).status_code == 422
    route = "/api/v1/uplift/benchmarks"
    params = {
        "artifact": "05/uplift_ranking_metrics.csv",
        "dataset": "Criteo 1M",
        "outcome": "conversion",
        "model": "s_lightgbm",
    }
    result = client.get(route, params=params).json()
    assert result["total"] == 1
    assert result["rows"][0]["policy"] == "s_lightgbm"
    assert client.get(route, params=params | {"model": "unknown"}).status_code == 404


def test_fixture_allocation_constraints_and_report(client):
    route = "/api/v1/policies/optimize"
    low = client.post(route, json={"budget": 60}).json()
    high = client.post(route, json={"budget": 120}).json()
    assert (low["selected"], high["selected"]) == (30, 60)
    assert low["allocation_fingerprint"] != high["allocation_fingerprint"]
    capped = client.post(route, json={"budget": 120, "capacity": 5}).json()
    assert capped["selected"] == 5 and capped["spend"] == 10 and capped["feasible"]
    free = client.post(
        route, json={"budget": 0, "cost": 0, "capacity": 240, "policy": "random"}
    ).json()
    assert free["selected"] == free["eligible"] < free["population"]
    expensive = client.post(route, json={"cost": 10}).json()
    assert expensive["selected"] == 0
    none = client.post(route, json={"policy": "none"}).json()
    assert none["selected"] == 0 and none["expected_roi"] is None
    f = client.post(route, json={"dataset": "fixture_F"}).json()
    assert f["identification"] == "not_identified"
    assert f["true_expected_synthetic_profit"] is None
    response = client.get("/api/v1/reports/business")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert "7,091.50" in response.text and "Session 05" in response.text
    assert "hypothetical" in response.text and "tau_true" not in response.text


def test_all_dashboard_pages_and_interactions(aggregates):
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30).run()
    for page in app.radio[0].options:
        app.radio[0].set_value(page).run()
        assert not app.exception, page
        assert app.title
    downloads = app.get("download_button")
    assert len(downloads) == 2
    app.radio[0].set_value("Experimentation lab").run()
    app.button[0].click().run()
    assert not app.exception
    assert any(m.label == "Absolute lift" and m.value == "2.00 pp" for m in app.metric)
    app.button[1].click().run()
    expected = binary_sample_size(0.1, absolute_effect=0.02)
    assert any(
        m.label == "Control required" and m.value == str(expected.control_n)
        for m in app.metric
    )
    app.radio[0].set_value("Business policy lab").run()
    assert app.metric[0].value == "60"
    app.number_input[0].set_value(60.0)
    app.button[0].click().run()
    assert not app.exception and app.metric[0].value == "30"
    aggregates.path("12/budget_profit_frontier.csv").unlink()
    app.run()
    assert not app.exception
    assert any("artifact unavailable" in warning.value for warning in app.warning)
