#!/usr/bin/env python3
"""Ingest provided numeric climate files and generate ML-ready data + charts."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
OFFICE_VALUE = f"{{{NS['office']}}}value"
OFFICE_DATE_VALUE = f"{{{NS['office']}}}date-value"
TABLE_REPEAT = f"{{{NS['table']}}}number-columns-repeated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified observations from sample numeric files and draw charts."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/DATA/Sayısallaştırılmış Veri"),
        help="Directory containing numeric input files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/sample"),
        help="Output directory for dataset and plots",
    )
    return parser.parse_args()


def to_float(v: object) -> float | None:
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


def safe_date(year: int, month: int, day: int) -> Optional[dt.datetime]:
    try:
        return dt.datetime(year=year, month=month, day=day)
    except ValueError:
        return None


def ingest_temp_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    hour_cols = []
    for c in df.columns[1:]:
        h = to_float(c)
        if h is not None and 1 <= int(h) <= 24:
            hour_cols.append(c)
    long = df.melt(id_vars=["date"], value_vars=hour_cols, var_name="hour", value_name="value")
    long["hour"] = long["hour"].apply(lambda x: int(to_float(x) or 0))
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["date", "value"])
    long["timestamp"] = long["date"] + pd.to_timedelta(long["hour"] - 1, unit="h")
    long["variable"] = "temp"
    long["unit"] = "degC"
    long["source_file"] = str(path)
    long["method"] = "xlsx_hourly_wide"
    long["confidence"] = 0.99
    long["raw_payload"] = long.apply(
        lambda r: json.dumps(
            {"date": r["date"].strftime("%Y-%m-%d"), "hour": int(r["hour"]), "value": float(r["value"])},
            ensure_ascii=False,
        ),
        axis=1,
    )
    return long[["timestamp", "variable", "value", "unit", "source_file", "method", "confidence", "raw_payload"]]


def ingest_humidity_xlsx(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    year_cols: list[tuple[int, int]] = []
    for col in range(raw.shape[1]):
        y = to_float(raw.iat[1, col] if 1 < raw.shape[0] else None)
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
            v = to_float(raw.iat[i, col])
            if v is None:
                continue
            ts = safe_date(year, month, day)
            if ts is None:
                continue
            recs.append(
                {
                    "timestamp": ts,
                    "variable": "humidity",
                    "value": v,
                    "unit": "pct",
                    "source_file": str(path),
                    "method": "xlsx_multi_year_wide",
                    "confidence": 0.96,
                    "raw_payload": json.dumps(
                        {"template_day": f"{month:02d}-{day:02d}", "year": year, "value": v},
                        ensure_ascii=False,
                    ),
                }
            )
    return pd.DataFrame(recs)


def ingest_precip_xlsx(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    year_cols: list[tuple[int, int]] = []
    seen_year_start = False
    for col in range(1, raw.shape[1]):
        y = to_float(raw.iat[0, col] if raw.shape[0] > 0 else None)
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
            v = to_float(raw.iat[i, col])
            if v is None:
                continue
            ts = safe_date(year, month, day)
            if ts is None:
                continue
            recs.append(
                {
                    "timestamp": ts,
                    "variable": "precip",
                    "value": v,
                    "unit": "mm",
                    "source_file": str(path),
                    "method": "xlsx_multi_year_wide",
                    "confidence": 0.95,
                    "raw_payload": json.dumps(
                        {"template_day": f"{month:02d}-{day:02d}", "year": year, "value": v},
                        ensure_ascii=False,
                    ),
                }
            )
    return pd.DataFrame(recs)


def iter_expanded_cells(row: ET.Element) -> Iterable[ET.Element]:
    for cell in row.findall("table:table-cell", NS):
        repeat = int(cell.attrib.get(TABLE_REPEAT, "1"))
        for _ in range(repeat):
            yield cell


def cell_text(cell: ET.Element) -> str:
    vals = []
    for p in cell.findall("text:p", NS):
        if p.text:
            vals.append(p.text.strip())
    return " ".join(v for v in vals if v)


def cell_float(cell: ET.Element) -> float | None:
    raw = cell.attrib.get(OFFICE_VALUE)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            return None
    return to_float(cell_text(cell))


def ingest_pressure_ods(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("content.xml")
    root = ET.parse(io.BytesIO(xml)).getroot()
    table = root.find(".//table:table", NS)
    if table is None:
        return pd.DataFrame(columns=["timestamp", "variable", "value", "unit", "source_file", "method", "confidence", "raw_payload"])

    rows = table.findall("table:table-row", NS)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "variable", "value", "unit", "source_file", "method", "confidence", "raw_payload"])

    header = list(iter_expanded_cells(rows[0]))
    years: list[int] = []
    for c in header[1:60]:
        y = cell_float(c)
        if y is None:
            if years:
                break
            continue
        yi = int(y)
        if 1800 <= yi <= 2100:
            years.append(yi)
        elif years:
            break

    recs: list[dict[str, object]] = []
    for row in rows[1:80]:
        cells = list(iter_expanded_cells(row))
        if not cells:
            continue
        day_val = cell_float(cells[0])
        if day_val is None:
            continue
        day = int(day_val)
        if day < 1 or day > 31:
            continue
        for j, year in enumerate(years, start=1):
            if j >= len(cells):
                break
            v = cell_float(cells[j])
            if v is None:
                continue
            ts = safe_date(year, 2, day)
            if ts is None:
                continue
            recs.append(
                {
                    "timestamp": ts,
                    "variable": "pressure",
                    "value": v,
                    "unit": "unknown",
                    "source_file": str(path),
                    "method": "ods_xml_wide",
                    "confidence": 0.9,
                    "raw_payload": json.dumps(
                        {"month": 2, "day": day, "year": year, "value": v},
                        ensure_ascii=False,
                    ),
                }
            )
    return pd.DataFrame(recs)


def qc_flag(variable: str, value: float) -> str:
    if pd.isna(value):
        return "missing"
    if variable == "temp" and not (-60 <= value <= 70):
        return "range_fail"
    if variable == "humidity" and not (0 <= value <= 100):
        return "range_fail"
    if variable == "precip" and value < 0:
        return "range_fail"
    return "ok"


def write_dataset(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp", "value"]).sort_values(["timestamp", "variable"])
    out["station_id"] = "KRDAE_KLIMA"
    out["is_missing"] = False
    out["qc_flag"] = [qc_flag(v, val) for v, val in zip(out["variable"], out["value"])]
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["day"] = out["timestamp"].dt.day
    out["hour"] = out["timestamp"].dt.hour
    cols = [
        "timestamp",
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
        "method",
        "raw_payload",
    ]
    out = out[cols].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / "observations_numeric.parquet"
    csv = out_dir / "observations_numeric.csv"
    out.to_parquet(pq, index=False)
    out.to_csv(csv, index=False)
    return pq, csv


def plot_charts(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["qc_flag"] == "ok"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    created: list[Path] = []

    monthly = (
        df.set_index("timestamp")
        .groupby("variable")["value"]
        .resample("MS")
        .mean()
        .reset_index()
        .sort_values(["variable", "timestamp"])
    )
    for var in sorted(monthly["variable"].unique()):
        sub = monthly[monthly["variable"] == var]
        n_pts = sub["value"].notna().sum()
        marker = "o" if n_pts <= 80 else None
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(sub["timestamp"], sub["value"], linewidth=1.4, marker=marker, markersize=2.8)
        ax.set_title(f"Aylik Ortalama - {var}")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Deger")
        ax.grid(alpha=0.25)
        p = out_dir / f"monthly_{var}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        created.append(p)

    annual = (
        df.set_index("timestamp")
        .groupby("variable")["value"]
        .resample("YS")
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for var in sorted(annual["variable"].unique()):
        sub = annual[annual["variable"] == var]
        ax.plot(sub["timestamp"].dt.year, sub["value"], marker="o", linewidth=1.2, label=var)
    ax.set_title("Yillik Ortalama Trendler")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Deger")
    ax.grid(alpha=0.25)
    ax.legend()
    p = out_dir / "annual_trends.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(p)

    pivot = monthly.pivot(index="timestamp", columns="variable", values="value")
    corr = pivot.corr(min_periods=12)
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=30, ha="right")
        ax.set_yticklabels(corr.index)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr.iat[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title("Aylik Korelasyon Matrisi")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        p = out_dir / "correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        created.append(p)

    return created


def main() -> None:
    args = parse_args()
    d = args.data_dir
    frames: list[pd.DataFrame] = []

    temp = d / "1987_Sıcaklık_Saat Başı.xlsx"
    hum = d / "Nem-1980-2014.xlsx"
    precip = d / "Yağış_1980-2019.xlsx"
    pressure_ods = d / "Basınç_Şubat_1984-2013.ods"

    if temp.exists():
        frames.append(ingest_temp_xlsx(temp))
    if hum.exists():
        frames.append(ingest_humidity_xlsx(hum))
    if precip.exists():
        frames.append(ingest_precip_xlsx(precip))
    if pressure_ods.exists():
        frames.append(ingest_pressure_ods(pressure_ods))

    if not frames:
        raise SystemExit(f"No known files found in {d}")

    combined = pd.concat(frames, ignore_index=True)
    pq, csv = write_dataset(combined, args.output_dir)
    df = pd.read_parquet(pq)
    charts = plot_charts(df, args.output_dir / "charts")

    summary = (
        df.groupby("variable")
        .agg(rows=("value", "size"), min_ts=("timestamp", "min"), max_ts=("timestamp", "max"))
        .reset_index()
    )
    print(f"Wrote dataset:\n- {pq}\n- {csv}\n")
    print("Variable summary:")
    print(summary.to_string(index=False))
    print("\nCharts:")
    for c in charts:
        print(f"- {c}")


if __name__ == "__main__":
    main()
