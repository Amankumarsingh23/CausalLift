import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from causallift.data.contract import DatasetSpec
from causallift.data.splits import allocate

SOURCE = "https://ailab.criteo.com/criteo-uplift-prediction-dataset/"
MIRROR = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz?download=true"
SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
RAW_NAME = "criteo-research-uplift-v2.1.csv.gz"
CRITEO = DatasetSpec(
    dataset_id="criteo_v2_1", source_type="experiment", treatment_column="treatment",
    outcome_column="conversion", feature_columns=tuple(f"f{i}" for i in range(12)),
    secondary_outcomes=("visit",), post_treatment_columns=("exposure",),
    provenance={"source": SOURCE, "publisher_copy": MIRROR, "license": "CC BY-NC-SA 4.0"},
    assumptions=("Pooled randomized assignment; released data were non-uniformly subsampled.",),
)
FIXTURE = replace(
    CRITEO, dataset_id="local_fixture", source_type="fixture",
    provenance={"source": "Handmade test fixture with Criteo-compatible schema"}, assumptions=(),
)


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_NAME
    if target.exists():
        if checksum(target) != SHA256:
            raise ValueError(f"Existing raw file fails publisher checksum: {target}")
        return target
    with NamedTemporaryFile(dir=raw_dir, suffix=".part", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with urlopen(MIRROR, timeout=30) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
        if checksum(path) != SHA256:
            raise ValueError("Download fails publisher checksum")
        # Publish without replacing an existing raw file, including on concurrent calls.
        target.hardlink_to(path)
    finally:
        path.unlink(missing_ok=True)
    return target


def ingest(path: Path, size: int, seed: int, *, fixture: bool = False, chunk_size: int = 200_000):
    """Profile every source row and keep a bounded uniform sample within each joint stratum."""
    if size < 1 or chunk_size < 1:
        raise ValueError("Subset and chunk sizes must be positive")
    fingerprint = checksum(path)
    spec = FIXTURE if fixture else CRITEO
    rng = np.random.default_rng(seed)
    dtype = dict.fromkeys(spec.required_columns, "float64")
    header = pd.read_csv(path, nrows=0).columns
    if set(header) != set(dtype):
        raise ValueError(f"Unexpected CSV header: {list(header)}")
    quality = {key: dict.fromkeys(header, 0) for key in ("missing", "invalid")}
    counts = np.zeros(4, dtype=np.int64)
    sums = np.zeros((2, 4), dtype=np.int64)
    pools = {}
    rows = 0
    for frame in pd.read_csv(path, chunksize=chunk_size, dtype=dtype):
        rows += len(frame)
        batch_quality = spec.quality(frame)
        for kind, totals in quality.items():
            for name in totals:
                totals[name] += batch_quality[kind][name]
        if any(sum(totals.values()) for totals in batch_quality.values()):
            continue
        frame = frame.astype(dict.fromkeys(spec.binary_columns, "int8"))
        for arm, group in frame.groupby("treatment"):
            sums[arm] += [len(group), group["conversion"].sum(), group["visit"].sum(), group["exposure"].sum()]
        labels = 2 * frame["treatment"] + frame["conversion"]
        frame["_priority"] = rng.random(len(frame))
        for label, group in frame.groupby(labels):
            counts[label] += len(group)
            previous = pools.get(label)
            candidates = group if previous is None else pd.concat([previous, group], ignore_index=True)
            pools[label] = candidates.nsmallest(size, "_priority")

    if not rows or any(sum(totals.values()) for totals in quality.values()):
        raise ValueError(f"Source validation failed ({rows} rows): {quality}")
    if not fixture and fingerprint == SHA256 and rows != 13_979_592:
        raise ValueError(f"Publisher archive row count differs from documentation: {rows}")
    quotas = allocate(min(rows, size), counts / rows)
    subset = pd.concat(
        [pools[label].iloc[:quotas[label]] for label in sorted(pools)], ignore_index=True,
    ).drop(columns="_priority")
    subset = subset.iloc[rng.permutation(len(subset))].reset_index(drop=True)
    spec.validate(subset)
    profile = {
        "source_rows": rows, "processed_rows": len(subset), "working_subset_rows": len(subset),
        "source_arm_summary": [
            {"treatment": arm, "rows": int(n), "proportion": float(n / rows),
             "conversion_rate": float(conv / n) if n else None,
             "visit_rate": float(visit / n) if n else None,
             "exposure_rate": float(exposure / n) if n else None}
            for arm, (n, conv, visit, exposure) in enumerate(sums)
        ],
        "sha256": fingerprint,
        "publisher_archive_verified": not fixture and fingerprint == SHA256,
        "source_file": path.name, "source_schema": list(header),
    }
    return subset, spec, profile, quality
