from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causallift.data.analytics import parquet_missing, parquet_summary
from causallift.data.contract import load_dataset
from causallift.data.criteo import CRITEO, FIXTURE, ingest
from causallift.data.splits import split_data
from causallift.simulation import simulate

FIXTURE_PATH = Path(__file__).parent / "fixtures/criteo_schema.csv"


@pytest.mark.parametrize("column", ["treatment", "conversion", "visit", "exposure", "split"])
def test_forbidden_criteo_features(column):
    with pytest.raises(ValueError, match="Forbidden"):
        replace(CRITEO, feature_columns=(*CRITEO.feature_columns, column))


def test_ground_truth_leakage():
    _, _, spec = simulate("F", 50)
    for column in (*spec.ground_truth_columns, "unit_id"):
        with pytest.raises(ValueError, match="Forbidden"):
            replace(spec, feature_columns=(*spec.feature_columns, column))


@pytest.mark.parametrize(("column", "value"), [
    ("treatment", 1.5), ("conversion", 2), ("visit", -1),
    ("exposure", np.nan), ("f0", np.inf), ("f0", "nonnumeric"),
])
def test_invalid_data(column, value):
    frame = pd.read_csv(FIXTURE_PATH)
    frame[column] = value
    with pytest.raises(ValueError, match=column):
        FIXTURE.validate(frame)


def test_invalid_schema():
    frame = pd.read_csv(FIXTURE_PATH).drop(columns="f0")
    with pytest.raises(ValueError, match="Missing required"):
        FIXTURE.validate(frame)


def test_ingest_keeps_duplicates_and_validates_before_cast(tmp_path):
    source = tmp_path / "input.csv"
    frame = pd.read_csv(FIXTURE_PATH)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(source, index=False)
    result, _, profile, _ = ingest(source, 100, 42, fixture=True, chunk_size=13)
    assert profile["source_rows"] == len(result) == 61
    assert result.duplicated().sum() == 1
    frame["treatment"] = frame["treatment"].astype(float)
    frame.loc[0, "treatment"] = 0.5
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Source validation failed"):
        ingest(source, 100, 42, fixture=True)


def test_ingest_sampling_independent_of_chunk_size():
    first, _, profile, _ = ingest(FIXTURE_PATH, 32, 42, fixture=True, chunk_size=13)
    second, _, _, _ = ingest(FIXTURE_PATH, 32, 42, fixture=True, chunk_size=60)
    pd.testing.assert_frame_equal(first, second)
    assert profile["processed_rows"] == 32
    assert first["conversion"].mean() == pytest.approx(0.2, abs=0.04)


def test_reproducible_partition_and_loader(tmp_path):
    frame, _, spec = simulate("B", 1200)
    split = split_data(frame, spec, 12)
    pd.testing.assert_series_equal(split, split_data(frame, spec, 12))
    assert not split.equals(split_data(frame, spec, 13))
    path = tmp_path / "dataset.parquet"
    frame.assign(split=split).to_parquet(path, index=False)
    parts = [load_dataset(path, spec, name) for name in ("train", "validation", "test")]
    assert sum(len(part) for part in parts) == len(frame)
    assert pd.concat(parts)["unit_id"].is_unique
    for part, expected in zip(parts, (720, 240, 240), strict=True):
        assert abs(len(part) - expected) <= 2
        assert set(part["treatment"]) == {0, 1}


def test_group_safety():
    frame, _, spec = simulate("D", 300)
    frame = pd.concat([frame, frame], ignore_index=True)
    result = frame.assign(split=split_data(frame, spec))
    assert result.groupby("unit_id")["split"].nunique().eq(1).all()
    assert result.groupby("split")["treatment"].nunique().eq(2).all()


def test_rare_outcome_reported():
    frame = pd.read_csv(FIXTURE_PATH)
    frame["conversion"] = 0
    frame.loc[0, "conversion"] = 1
    with pytest.warns(UserWarning, match="lacks a treatment/outcome"):
        result = split_data(frame, FIXTURE)
    assert result.nunique() == 3


def test_infeasible_arm_coverage():
    frame = pd.read_csv(FIXTURE_PATH)
    frame["treatment"] = 0
    frame.loc[:1, "treatment"] = 1
    with pytest.warns(UserWarning, match="lacked arm coverage"), pytest.raises(ValueError, match="both treatment arms"):
        split_data(frame, FIXTURE)


def test_duckdb_matches_python(tmp_path):
    frame = pd.read_csv(FIXTURE_PATH)
    path = tmp_path / "analytics.parquet"
    frame.to_parquet(path, index=False)
    sql = parquet_summary(path, FIXTURE)
    python = frame.groupby("treatment").agg(
        rows=("treatment", "size"), conversion_rate=("conversion", "mean"),
        visit_rate=("visit", "mean"),
    ).reset_index()
    pd.testing.assert_frame_equal(sql, python, check_dtype=False)
    assert parquet_missing(path, tuple(frame.columns)) == frame.isna().sum().to_dict()
