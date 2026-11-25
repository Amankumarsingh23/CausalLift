"""Optional local MLflow tracking for one executed fixture policy."""

from pathlib import Path


def log_policy(result, destination):
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    path = Path(destination).resolve()
    path.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=path.as_uri())
    name = "CausalLift fixture policy"
    experiment = client.get_experiment_by_name(name)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(name)
    )
    run = client.create_run(
        experiment_id,
        tags={
            "dataset": result.dataset,
            "split": result.split,
            "model": result.model,
            "identification": result.identification,
            "fixture_sha256": result.fixture_fingerprint,
            "allocation_sha256": result.allocation_fingerprint,
            "artifact_reference": "artifacts/session_13/demo_run_manifest.json",
        },
    )
    params = [
        Param(key, str(value)) for key, value in result.economics.model_dump().items()
    ]
    params.append(Param("seed", str(result.seed)))
    metrics = [
        Metric(key, value, run.info.start_time, 0)
        for key, value in {
            "selected": result.selected,
            "spend": result.spend,
            "expected_profit": result.expected_profit,
        }.items()
    ]
    client.log_batch(run.info.run_id, params=params, metrics=metrics)
    client.set_terminated(run.info.run_id)
    return run.info.run_id
