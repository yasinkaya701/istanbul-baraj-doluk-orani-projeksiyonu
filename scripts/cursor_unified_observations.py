#!/usr/bin/env python3
"""Cursor unified observations: merge numeric and trace datasets into one canonical table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cursor: merge numeric and trace observations into a unified canonical dataset."
    )
    parser.add_argument(
        "--numeric-parquet",
        type=Path,
        default=Path("output/cursor_sample/cursor_observations_numeric.parquet"),
        help="Parquet file from cursor_ingest_numeric.py",
    )
    parser.add_argument(
        "--trace-parquet",
        type=Path,
        default=Path("output/cursor_sample/cursor_observations_traces.parquet"),
        help="Parquet file from cursor_ingest_traces.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cursor_forecast_package"),
        help="Directory for unified observations outputs.",
    )
    return parser.parse_args()


def _qc_range(variable: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if variable == "humidity" and not (0 <= value <= 100):
        return "range_fail"
    if variable == "temp" and not (-60 <= value <= 70):
        return "range_fail"
    if variable == "pressure" and not (0 <= value <= 2000):
        return "range_fail"
    if variable == "precip" and value < 0:
        return "range_fail"
    return "ok"


def build_unified(numeric_path: Path, trace_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if numeric_path.exists():
        frames.append(pd.read_parquet(numeric_path))
    if trace_path.exists():
        frames.append(pd.read_parquet(trace_path))

    if not frames:
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
                "qc_flag",
                "is_missing",
                "year",
                "month",
                "day",
                "hour",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["variable"] = df["variable"].astype(str)
    df = df.dropna(subset=["timestamp", "value", "variable"])

    df["qc_flag"] = [_qc_range(v, val) for v, val in zip(df["variable"], df["value"])]
    df["is_missing"] = False
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour

    return df.sort_values(["timestamp", "variable", "source_kind", "source_file"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    unified = build_unified(args.numeric_parquet, args.trace_parquet)
    parquet_path = out_dir / "cursor_observations_with_graph.parquet"
    csv_path = out_dir / "cursor_observations_with_graph.csv"
    unified.to_parquet(parquet_path, index=False)
    unified.to_csv(csv_path, index=False)

    if unified.empty:
        print("Unified dataset is empty (no numeric or trace inputs found).")
        return

    src_summary = (
        unified.groupby(["variable", "source_kind"])
        .agg(rows=("value", "size"), min_ts=("timestamp", "min"), max_ts=("timestamp", "max"))
        .reset_index()
    )
    print(f"Wrote unified observations:\n- {parquet_path}\n- {csv_path}\n")
    print("Source summary:")
    print(src_summary.to_string(index=False))


if __name__ == "__main__":
    main()

