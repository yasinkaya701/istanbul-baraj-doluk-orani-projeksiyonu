#!/usr/bin/env python3
"""Cursor numeric ingest: read climate spreadsheets into a canonical observations table.

Bu script mevcut projedeki hiçbir dosyayı değiştirmez. Yalnızca verilen
sayısal iklim dosyalarını okuyup aşağıdaki kanonik şemaya dönüştürür:

timestamp (datetime64[ns])
variable  (str: "temp" | "humidity" | "precip" | "pressure")
value     (float)
unit      (str)
station_id (str)
source_file (str)
source_kind (str: e.g. "numeric")
method      (str)
confidence  (float)

Çıktılar varsayılan olarak `output/cursor_sample/` altına yazılır.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cursor: ingest numeric climate files into a canonical observations dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DATA/Sayısallaştırılmış Veri"),
        help="Directory containing numeric input files (Excel/ODS).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cursor_sample"),
        help="Directory for cursor numeric observations outputs.",
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default="KRDAE_KLIMA",
        help="Station identifier used in the canonical table.",
    )
    # Optional explicit file overrides (useful for future datasets)
    parser.add_argument(
        "--temp-file",
        type=Path,
        help="Path to temperature spreadsheet (overrides default name lookup).",
    )
    parser.add_argument(
        "--humidity-file",
        type=Path,
        help="Path to humidity spreadsheet (overrides default name lookup).",
    )
    parser.add_argument(
        "--precip-file",
        type=Path,
        help="Path to precipitation spreadsheet (overrides default name lookup).",
    )
    parser.add_argument(
        "--pressure-file",
        type=Path,
        help="Path to pressure spreadsheet (overrides default name lookup).",
    )
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


def _safe_date(year: int, month: int, day: int) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def _default_temp_path(data_dir: Path) -> Path:
    return data_dir / "1987_Sıcaklık_Saat Başı.xlsx"


def _default_humidity_path(data_dir: Path) -> Path:
    return data_dir / "Nem-1980-2014.xlsx"


def _default_precip_path(data_dir: Path) -> Path:
    return data_dir / "Yağış_1980-2019.xlsx"


def _default_pressure_path(data_dir: Path) -> Path:
    return data_dir / "Basınç_Şubat_1984-2013.ods"


def ingest_temp_xlsx(path: Path, station_id: str) -> pd.DataFrame:
    """Temperature: hourly wide Excel -> long canonical form."""
    df = pd.read_excel(path)
    if df.empty:
        return pd.DataFrame()

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    hour_cols: list[str] = []
    for c in df.columns[1:]:
        h = _to_float(c)
        if h is not None and 1 <= int(h) <= 24:
            hour_cols.append(c)

    if not hour_cols:
        return pd.DataFrame()

    long = df.melt(id_vars=["date"], value_vars=hour_cols, var_name="hour", value_name="value")
    long["hour"] = long["hour"].apply(lambda x: int(_to_float(x) or 0))
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["date", "value"])
    long["timestamp"] = long["date"] + pd.to_timedelta(long["hour"] - 1, unit="h")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(long["timestamp"], errors="coerce"),
            "variable": "temp",
            "value": pd.to_numeric(long["value"], errors="coerce"),
            "unit": "degC",
            "station_id": station_id,
            "source_file": str(path),
            "source_kind": "numeric",
            "method": "xlsx_hourly_wide",
            "confidence": 0.99,
        }
    )
    return out.dropna(subset=["timestamp", "value"])


def ingest_humidity_xlsx(path: Path, station_id: str) -> pd.DataFrame:
    """Humidity: multi-year wide Excel -> daily canonical form."""
    raw = pd.read_excel(path, header=None)
    if raw.empty:
        return pd.DataFrame()

    year_cols: list[tuple[int, int]] = []
    for col in range(raw.shape[1]):
        y = _to_float(raw.iat[1, col] if 1 < raw.shape[0] else None)
        if y is not None and 1800 <= int(y) <= 2100:
            year_cols.append((col, int(y)))

    date_template = pd.to_datetime(raw.iloc[3:, 0], errors="coerce")

    recs: list[dict[str, object]] = []
    for i, t in enumerate(date_template, start=3):
        if pd.isna(t):
            continue
        month = int(t.month)
        day = int(t.day)
        for col, year in year_cols:
            v = _to_float(raw.iat[i, col])
            if v is None:
                continue
            ts = _safe_date(year, month, day)
            if ts is None:
                continue
            recs.append(
                {
                    "timestamp": ts,
                    "variable": "humidity",
                    "value": v,
                    "unit": "pct",
                    "station_id": station_id,
                    "source_file": str(path),
                    "source_kind": "numeric",
                    "method": "xlsx_multi_year_wide",
                    "confidence": 0.96,
                }
            )
    return pd.DataFrame(recs)


def ingest_precip_xlsx(path: Path, station_id: str) -> pd.DataFrame:
    """Precipitation: multi-year wide Excel -> daily canonical form."""
    raw = pd.read_excel(path, header=None)
    if raw.empty:
        return pd.DataFrame()

    year_cols: list[tuple[int, int]] = []
    seen_year_start = False
    for col in range(1, raw.shape[1]):
        y = _to_float(raw.iat[0, col] if raw.shape[0] > 0 else None)
        if y is not None and 1800 <= int(y) <= 2100:
            year_cols.append((col, int(y)))
            seen_year_start = True
            continue
        if seen_year_start:
            break

    date_template = pd.to_datetime(raw.iloc[1:, 0], errors="coerce")

    recs: list[dict[str, object]] = []
    for i, t in enumerate(date_template, start=1):
        if pd.isna(t):
            continue
        month = int(t.month)
        day = int(t.day)
        for col, year in year_cols:
            v = _to_float(raw.iat[i, col])
            if v is None:
                continue
            ts = _safe_date(year, month, day)
            if ts is None:
                continue
            recs.append(
                {
                    "timestamp": ts,
                    "variable": "precip",
                    "value": v,
                    "unit": "mm",
                    "station_id": station_id,
                    "source_file": str(path),
                    "source_kind": "numeric",
                    "method": "xlsx_multi_year_wide",
                    "confidence": 0.95,
                }
            )
    return pd.DataFrame(recs)


def ingest_pressure_ods(path: Path, station_id: str) -> pd.DataFrame:
    """Pressure: reuse `extract_ods_hourly.py` outputs once available.

Bu fonksiyon şimdilik yer tutucu; doğrudan ODS parse etmek yerine
gelecekte üretilecek ara CSV'leri okumaya uyarlanabilir.
"""
    # Not implemented yet; return empty frame to avoid breaking the pipeline.
    _ = (path, station_id)
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


def build_canonical_dataset(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(list(frames), ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value", "variable"])
    return df.sort_values(["timestamp", "variable", "source_file"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    temp_path = args.temp_file or _default_temp_path(data_dir)
    if temp_path.exists():
        frames.append(ingest_temp_xlsx(temp_path, args.station_id))

    humidity_path = args.humidity_file or _default_humidity_path(data_dir)
    if humidity_path.exists():
        frames.append(ingest_humidity_xlsx(humidity_path, args.station_id))

    precip_path = args.precip_file or _default_precip_path(data_dir)
    if precip_path.exists():
        frames.append(ingest_precip_xlsx(precip_path, args.station_id))

    pressure_path = args.pressure_file or _default_pressure_path(data_dir)
    if pressure_path.exists():
        frames.append(ingest_pressure_ods(pressure_path, args.station_id))

    if not frames:
        raise SystemExit(f"No known numeric input files found under {data_dir}")

    canonical = build_canonical_dataset(frames)
    if canonical.empty:
        raise SystemExit("Numeric files were found but produced no valid rows.")

    parquet_path = out_dir / "cursor_observations_numeric.parquet"
    csv_path = out_dir / "cursor_observations_numeric.csv"
    canonical.to_parquet(parquet_path, index=False)
    canonical.to_csv(csv_path, index=False)

    summary = (
        canonical.groupby("variable")
        .agg(rows=("value", "size"), min_ts=("timestamp", "min"), max_ts=("timestamp", "max"))
        .reset_index()
    )
    print(f"Wrote canonical numeric observations:\n- {parquet_path}\n- {csv_path}\n")
    print("Variable summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

