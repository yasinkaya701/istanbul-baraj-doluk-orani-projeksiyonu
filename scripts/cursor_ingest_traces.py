#!/usr/bin/env python3
"""Cursor trace ingest: read digitized graph-paper traces into canonical observations.

Bu script, graf kağıdından sayısallaştırılmış iz CSV'lerini okuyup
kanonik şemaya dönüştürür:

timestamp, variable, value, unit, station_id, source_file, source_kind, method, confidence

Varsayılan olarak yalnızca nem izlerini (*_humidity_trace.csv) okur,
ama parametrelerle diğer değişkenler için de genişletilebilir.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cursor: ingest digitized graph-paper traces into a canonical observations dataset."
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing *_humidity_trace.csv (and similar) files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cursor_sample"),
        help="Directory for cursor trace observations outputs.",
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default="KRDAE_KLIMA",
        help="Station identifier used in the canonical table.",
    )
    # Future extension: generic mappings for other variables.
    return parser.parse_args()


def _to_float(v: object) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date_from_filename(path: Path) -> pd.Timestamp | None:
    """Extract base date like YYYY_MM_DD from filename."""
    m = re.search(r"(19|20)\d{2}[_-](\d{2})[_-](\d{2})", path.name)
    if not m:
        return None
    y = int(path.name[m.start() : m.start() + 4])
    mm = int(m.group(2))
    dd = int(m.group(3))
    try:
        return pd.Timestamp(year=y, month=mm, day=dd)
    except ValueError:
        return None


def _load_humidity_trace(path: Path, station_id: str) -> pd.DataFrame:
    """Read a single *_humidity_trace.csv into canonical form."""
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    ts = pd.to_datetime(df.get("timestamp", pd.Series(dtype=object)), errors="coerce")

    # If timestamp missing, try reconstruct from elapsed_hour + filename date.
    if ts.isna().all():
        base_date = _parse_date_from_filename(path)
        elapsed = pd.to_numeric(df.get("elapsed_hour"), errors="coerce")
        if base_date is not None and elapsed.notna().any():
            ts = base_date + pd.to_timedelta(8, unit="h") + pd.to_timedelta(elapsed.fillna(0), unit="h")

    value = pd.to_numeric(df.get("humidity_pct"), errors="coerce")

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "variable": "humidity",
            "value": value,
            "unit": "pct",
            "station_id": station_id,
            "source_file": str(path),
            "source_kind": "graph_paper",
            "method": "trace_digitize",
            "confidence": 0.78,
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["timestamp", "value"])


def build_canonical_traces(trace_dir: Path, station_id: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for p in sorted(trace_dir.glob("*_humidity_trace.csv")):
        rows.append(_load_humidity_trace(p, station_id))
    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "variable",
                "value",
                "unit",
                "station_id",
                "source_file",
                "source_kind",
                "method",
                "confidence",
            ]
        )
    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value", "variable"])
    return df.sort_values(["timestamp", "variable", "source_file"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = build_canonical_traces(args.trace_dir, args.station_id)
    parquet_path = out_dir / "cursor_observations_traces.parquet"
    csv_path = out_dir / "cursor_observations_traces.csv"
    canonical.to_parquet(parquet_path, index=False)
    canonical.to_csv(csv_path, index=False)

    if canonical.empty:
        print(f"No *_humidity_trace.csv files found under {args.trace_dir}, wrote empty tables.")
        return

    summary = (
        canonical.groupby("variable")
        .agg(rows=("value", "size"), min_ts=("timestamp", "min"), max_ts=("timestamp", "max"))
        .reset_index()
    )
    print(f"Wrote canonical trace observations:\n- {parquet_path}\n- {csv_path}\n")
    print("Variable summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

