#!/usr/bin/env python3
"""Build a loss-aware canonical ML dataset from extracted climate CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create canonical observations.parquet/csv for ML."
    )
    parser.add_argument(
        "--temp-csv",
        type=Path,
        action="append",
        default=[],
        help="Hourly temperature CSV(s) from extract_ods_hourly.py",
    )
    parser.add_argument(
        "--humidity-csv",
        type=Path,
        action="append",
        default=[],
        help="Humidity trace CSV(s) from digitize_humidity_tif.py",
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default="KRDAE_KLIMA",
        help="Station identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ml"),
        help="Output directory",
    )
    return parser.parse_args()


def _qc_range(variable: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if variable == "humidity" and not (0 <= value <= 100):
        return "range_fail"
    if variable == "temp" and not (-60 <= value <= 70):
        return "range_fail"
    if variable == "pressure" and not (850 <= value <= 1100):
        return "range_fail"
    return "ok"


def _to_iso(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return ts.isoformat(timespec="minutes")


def _full_hour_grid(ts: pd.Series) -> pd.DatetimeIndex:
    t = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
    if t.empty:
        return pd.DatetimeIndex([])
    start = t.min().floor("h")
    end = t.max().floor("h")
    return pd.date_range(start, end, freq="h")


def from_temp_csv(path: Path, station_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns and df["timestamp"].astype(str).str.strip().ne("").any():
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    elif {"date", "hour"}.issubset(df.columns):
        base = pd.to_datetime(df["date"], errors="coerce")
        ts = base + pd.to_timedelta(pd.to_numeric(df["hour"], errors="coerce") - 1, unit="h")
    else:
        raise ValueError(f"{path}: no timestamp/date-hour columns")

    raw = pd.DataFrame(
        {
            "timestamp": ts,
            "station_id": station_id,
            "variable": "temp",
            "value": pd.to_numeric(df.get("temp_c"), errors="coerce"),
            "unit": "degC",
            "source_file": str(path),
            "source_kind": "table_hourly",
            "method": "ods_xml_parse",
            "confidence": 0.99,
            "source_row_idx": range(len(df)),
            "raw_payload": [json.dumps(row, ensure_ascii=False) for row in df.to_dict("records")],
        }
    )

    # Loss-aware: include every hour in observed range, flag missing if absent.
    grid = pd.DataFrame({"timestamp": _full_hour_grid(raw["timestamp"])})
    merged = grid.merge(raw, on="timestamp", how="left")
    merged["station_id"] = merged["station_id"].fillna(station_id)
    merged["variable"] = merged["variable"].fillna("temp")
    merged["unit"] = merged["unit"].fillna("degC")
    merged["source_file"] = merged["source_file"].fillna(str(path))
    merged["source_kind"] = merged["source_kind"].fillna("table_hourly")
    merged["method"] = merged["method"].fillna("ods_xml_parse")
    merged["confidence"] = merged["confidence"].fillna(0.0)
    merged["source_row_idx"] = merged["source_row_idx"].fillna(-1).astype(int)
    merged["raw_payload"] = merged["raw_payload"].fillna("")
    return merged


def from_humidity_csv(path: Path, station_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.get("timestamp", pd.Series(dtype=object)), errors="coerce")
    value = pd.to_numeric(df.get("humidity_pct"), errors="coerce")
    raw = pd.DataFrame(
        {
            "timestamp": ts,
            "station_id": station_id,
            "variable": "humidity",
            "value": value,
            "unit": "pct",
            "source_file": str(path),
            "source_kind": "chart_trace",
            "method": "color_mask_trace",
            "confidence": 0.75,
            "source_row_idx": range(len(df)),
            "x_px": pd.to_numeric(df.get("x_px"), errors="coerce"),
            "y_px": pd.to_numeric(df.get("y_px"), errors="coerce"),
            "hour_of_day": pd.to_numeric(df.get("hour_of_day"), errors="coerce"),
            "elapsed_hour": pd.to_numeric(df.get("elapsed_hour"), errors="coerce"),
            "raw_payload": [json.dumps(row, ensure_ascii=False) for row in df.to_dict("records")],
        }
    )
    return raw


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["is_missing"] = out["value"].isna()
    out["qc_flag"] = [
        _qc_range(v, val) for v, val in zip(out["variable"].astype(str), out["value"])
    ]
    out["timestamp_iso"] = [_to_iso(ts) for ts in out["timestamp"]]
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["day"] = out["timestamp"].dt.day
    out["hour"] = out["timestamp"].dt.hour
    cols = [
        "timestamp",
        "timestamp_iso",
        "year",
        "month",
        "day",
        "hour",
        "station_id",
        "variable",
        "value",
        "unit",
        "is_missing",
        "qc_flag",
        "confidence",
        "source_file",
        "source_kind",
        "method",
        "source_row_idx",
        "x_px",
        "y_px",
        "hour_of_day",
        "elapsed_hour",
        "raw_payload",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols].sort_values(["timestamp", "variable", "source_file"], na_position="last")
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for p in args.temp_csv:
        if p.exists():
            frames.append(from_temp_csv(p, args.station_id))
    for p in args.humidity_csv:
        if p.exists():
            frames.append(from_humidity_csv(p, args.station_id))
    if not frames:
        raise SystemExit("No input files found.")

    out = finalize(pd.concat(frames, ignore_index=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq_path = args.output_dir / "observations.parquet"
    csv_path = args.output_dir / "observations.csv"
    out.to_parquet(pq_path, index=False)
    out.to_csv(csv_path, index=False)

    # Model-ready wide views per variable (monthly means).
    monthly = (
        out.dropna(subset=["timestamp"])
        .assign(value_num=pd.to_numeric(out["value"], errors="coerce"))
        .dropna(subset=["value_num"])
        .set_index("timestamp")
        .groupby("variable")["value_num"]
        .resample("MS")
        .mean()
        .reset_index()
    )
    monthly.to_parquet(args.output_dir / "monthly_features.parquet", index=False)
    monthly.to_csv(args.output_dir / "monthly_features.csv", index=False)

    print(f"Wrote {len(out)} rows:")
    print(f"- {pq_path}")
    print(f"- {csv_path}")
    print(f"- {args.output_dir / 'monthly_features.parquet'}")
    print(f"- {args.output_dir / 'monthly_features.csv'}")


if __name__ == "__main__":
    main()

