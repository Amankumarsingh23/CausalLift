import csv as csv_parser
import gzip
from pathlib import Path

import duckdb
import pandas as pd

from causallift.data.contract import DatasetSpec


def identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def binary_arm_counts(path: Path, spec: DatasetSpec, *, csv: bool = False) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return control and treated tuples of row count and exact event totals."""
    treatment = identifier(spec.treatment_column)
    outcome_names = (spec.outcome_column, *spec.secondary_outcomes)
    outcomes = tuple(map(identifier, outcome_names))
    read = "read_parquet(?)"
    if csv:
        # Explicit types avoid inference and allow projection of the three binary fields.
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as file:
            columns = next(csv_parser.reader(file), [])
        required = {spec.treatment_column, *outcome_names}
        if len(set(columns)) != len(columns) or not required.issubset(columns):
            raise ValueError("CSV header must contain unique assignment and outcome columns")
        schema = {name: "DOUBLE" if name == spec.treatment_column or name in outcome_names else "VARCHAR"
                  for name in columns}
        read = "read_csv(?, header=true, auto_detect=false, delim=',', columns=?)"
    bad = " OR ".join(f"{name} IS NULL OR {name} NOT IN (0, 1)" for name in outcomes)
    totals = ", ".join(f"count(*) FILTER (WHERE {name} = 1)" for name in outcomes)
    with duckdb.connect() as db:
        rows = db.execute(
            f"SELECT {treatment}, count(*), count(*) FILTER (WHERE {bad}), {totals} "
            f"FROM {read} GROUP BY {treatment} ORDER BY {treatment}",
            [str(path), schema] if csv else [str(path)],
        ).fetchall()
    if [row[0] for row in rows] != [0, 1]:
        raise ValueError("Binary aggregation requires both arms and valid 0/1 treatment labels")
    if any(row[2] for row in rows):
        raise ValueError("Binary aggregation found missing or invalid outcomes")
    control, treated = [(row[1], *row[3:]) for row in rows]
    return control, treated


def parquet_summary(path: Path, spec: DatasetSpec, *, by_split: bool = False) -> pd.DataFrame:
    treatment = identifier(spec.treatment_column)
    rates = ", ".join(
        f"avg({identifier(name)}) AS {identifier(name + '_rate')}"
        for name in (spec.outcome_column, *spec.secondary_outcomes)
    )
    grouping = f"split, {treatment}" if by_split else treatment
    with duckdb.connect() as db:
        return db.execute(
            f"SELECT {grouping}, count(*) AS rows, {rates} FROM read_parquet(?) "
            f"GROUP BY {grouping} ORDER BY {grouping}",
            [str(path)],
        ).df()


def parquet_missing(path: Path, columns: tuple[str, ...]) -> dict[str, int]:
    counts = ", ".join(
        f"count(*) - count({identifier(name)}) AS {identifier(name)}" for name in columns
    )
    with duckdb.connect() as db:
        values = db.execute(f"SELECT {counts} FROM read_parquet(?)", [str(path)]).fetchone()
    return dict(zip(columns, values, strict=True))


def scenario_summary(observed: Path, truth: Path) -> dict:
    with duckdb.connect() as db:
        result = db.execute(
            """
            SELECT count(*) AS rows, avg(treatment) AS treatment_proportion,
                   avg(outcome) AS outcome_prevalence,
                   avg(tau_true) AS true_ate,
                   avg(tau_true) FILTER (WHERE treatment = 1) AS true_att,
                   avg(outcome) FILTER (WHERE treatment = 1) -
                     avg(outcome) FILTER (WHERE treatment = 0) AS naive_difference,
                   min(tau_true) AS cate_min, max(tau_true) AS cate_max
            FROM read_parquet(?) JOIN read_parquet(?) USING (unit_id)
            """,
            [str(observed), str(truth)],
        ).df()
    return result.iloc[0].to_dict() | {"rows": int(result.iloc[0]["rows"])}
