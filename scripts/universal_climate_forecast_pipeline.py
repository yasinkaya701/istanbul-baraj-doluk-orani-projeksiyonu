#!/usr/bin/env python3
"""Universal climate data ingestion + forecasting pipeline.

This script is designed to work on heterogeneous datasets:
- csv / parquet
- xlsx / xls / ods (including wide calendar-like tables)
- optional graph-paper trace csv files

It outputs:
1) unified observations
2) parse report
3) per-variable, per-frequency historical datasets
4) per-variable, per-frequency forecasts + charts
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


FREQ_CONFIG = {
    "hourly": {"freq": "1h", "horizon": 24 * 30, "seasonal_period": 24, "title": "Saatlik"},
    "daily": {"freq": "1D", "horizon": 365, "seasonal_period": 7, "title": "Gunluk"},
    "monthly": {"freq": "MS", "horizon": 24, "seasonal_period": 12, "title": "Aylik"},
    "yearly": {"freq": "YS", "horizon": 10, "seasonal_period": 5, "title": "Yillik"},
}

VAR_KEYWORDS = {
    "temp": ["temp", "temperature", "sicak", "sıcak", "t2m"],
    "humidity": ["humidity", "hum", "nem", "rh", "relative"],
    "pressure": ["pressure", "press", "basinc", "basınc", "hpa"],
    "precip": ["precip", "rain", "yagis", "yağis", "mm"],
}

VAR_UNITS = {
    "temp": "degC",
    "humidity": "pct",
    "pressure": "unknown",
    "precip": "mm",
}

FILE_EXTS = {".csv", ".parquet", ".xlsx", ".xls", ".ods"}

MONTH_KEYWORDS = {
    1: ["ocak", "january", "jan", "oc"],
    2: ["subat", "şubat", "february", "feb", "sub"],
    3: ["mart", "march", "mar"],
    4: ["nisan", "april", "apr", "nis"],
    5: ["mayis", "mayıs", "may"],
    6: ["haziran", "june", "jun", "haz"],
    7: ["temmuz", "july", "jul", "tem"],
    8: ["agustos", "ağustos", "august", "aug", "agu"],
    9: ["eylul", "eylül", "september", "sep", "eyl"],
    10: ["ekim", "october", "oct", "eki"],
    11: ["kasim", "kasım", "november", "nov", "kas"],
    12: ["aralik", "aralık", "december", "dec", "ara"],
}

NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
OFFICE_VALUE = f"{{{NS['office']}}}value"
TABLE_REPEAT = f"{{{NS['table']}}}number-columns-repeated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal climate ingestion + forecasting.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory (recursive scan).")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Optional dir containing graph-paper trace csv files (e.g., *_trace.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/universal_forecast"),
        help="Output directory.",
    )
    parser.add_argument("--station-id", type=str, default="KRDAE_KLIMA", help="Station id.")
    parser.add_argument(
        "--freqs",
        type=str,
        default="hourly,daily,monthly,yearly",
        help="Comma-separated forecast frequencies.",
    )
    return parser.parse_args()


def normalize_text(s: object) -> str:
    t = str(s).strip().lower()
    tr_map = str.maketrans("çğıöşüı", "cgiosui")
    t = t.translate(tr_map)
    return re.sub(r"\s+", " ", t)


def infer_variable(*texts: object) -> Optional[str]:
    hay = " ".join(normalize_text(x) for x in texts if x is not None)
    for var, keys in VAR_KEYWORDS.items():
        if any(k in hay for k in keys):
            return var
    return None


def infer_month(*texts: object) -> Optional[int]:
    hay = " ".join(normalize_text(x) for x in texts if x is not None)
    for mm, keys in MONTH_KEYWORDS.items():
        if any(k in hay for k in keys):
            return mm
    return None


def to_float(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        if pd.isna(v):
            return None
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def list_files(input_dir: Path) -> list[Path]:
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in FILE_EXTS:
            files.append(p)
    return sorted(files)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cols = []
    for i, c in enumerate(out.columns):
        cc = str(c).strip()
        if not cc or cc.lower().startswith("unnamed"):
            cc = f"col_{i}"
        cols.append(cc)
    out.columns = cols
    return out


def read_ods_xml(path: Path) -> list[tuple[str, pd.DataFrame]]:
    result: list[tuple[str, pd.DataFrame]] = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("content.xml")
    root = ET.parse(io.BytesIO(xml)).getroot()

    for table in root.findall(".//table:table", NS):
        name = table.attrib.get(f"{{{NS['table']}}}name", "Sheet")
        matrix: list[list[object]] = []
        for row in table.findall("table:table-row", NS):
            vals: list[object] = []
            for cell in row.findall("table:table-cell", NS):
                repeat = int(cell.attrib.get(TABLE_REPEAT, "1"))
                val = cell.attrib.get(OFFICE_VALUE)
                if val is None:
                    parts = []
                    for p in cell.findall("text:p", NS):
                        if p.text:
                            parts.append(p.text.strip())
                    val = " ".join(parts) if parts else None
                for _ in range(repeat):
                    vals.append(val)
            # trim trailing empty cells
            while vals and (vals[-1] is None or str(vals[-1]).strip() == ""):
                vals.pop()
            if vals:
                matrix.append(vals)
            if len(matrix) > 40000:
                break
        if not matrix:
            continue
        max_w = max(len(r) for r in matrix)
        matrix = [r + [None] * (max_w - len(r)) for r in matrix]
        df = pd.DataFrame(matrix)
        result.append((name, df))
    return result


def read_tabular_file(path: Path) -> list[tuple[str, pd.DataFrame]]:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return [("parquet", pd.read_parquet(path))]
    if ext == ".csv":
        try:
            return [("csv", pd.read_csv(path))]
        except UnicodeDecodeError:
            return [("csv", pd.read_csv(path, encoding="latin1"))]
    if ext in {".xlsx", ".xls"}:
        try:
            xls = pd.ExcelFile(path)
            out = []
            for s in xls.sheet_names:
                out.append((s, pd.read_excel(path, sheet_name=s, header=None)))
            return out
        except Exception:
            try:
                return [("excel", pd.read_excel(path))]
            except Exception:
                return []
    if ext == ".ods":
        try:
            xls = pd.ExcelFile(path, engine="odf")
            out = []
            for s in xls.sheet_names:
                out.append((s, pd.read_excel(path, sheet_name=s, engine="odf", header=None)))
            return out
        except Exception:
            try:
                return read_ods_xml(path)
            except Exception:
                return []
    return []


def detect_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(k in normalize_text(c) for k in ["date", "tarih", "time", "timestamp"])]
    cols = candidates if candidates else list(df.columns)
    best_col = None
    best_score = 0.0
    sample_n = min(400, len(df))
    for c in cols:
        s = pd.to_datetime(df[c], errors="coerce")
        score = float(s.iloc[:sample_n].notna().mean()) if sample_n else 0.0
        if score > best_score:
            best_col = c
            best_score = score
    return best_col if best_score >= 0.55 else None


def detect_hour_cols(columns: Iterable[object]) -> list[str]:
    out = []
    for c in columns:
        f = to_float(c)
        if f is None:
            continue
        hh = int(f)
        if 0 <= hh <= 24 and abs(f - hh) < 1e-6:
            out.append(str(c))
    return out


def find_year_header_row(df: pd.DataFrame) -> Optional[int]:
    max_row = min(12, len(df))
    best_r = None
    best_count = 0
    for r in range(max_row):
        vals = [to_float(x) for x in df.iloc[r].tolist()]
        years = [int(v) for v in vals if v is not None and 1800 <= int(v) <= 2100]
        cnt = len(years)
        if cnt > best_count:
            best_count = cnt
            best_r = r
    return best_r if best_count >= 5 else None


def safe_date(year: int, month: int, day: int) -> Optional[pd.Timestamp]:
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return None


def extract_from_df(df_in: pd.DataFrame, file_path: Path, sheet_name: str, station_id: str) -> tuple[pd.DataFrame, dict[str, object]]:
    df = clean_df(df_in)
    file_var = infer_variable(file_path.name, sheet_name)
    file_month = infer_month(file_path.name, sheet_name)
    method = "unknown"
    confidence = 0.7
    r_year = find_year_header_row(df)

    # Pattern 0: already unified style
    cols_norm = {normalize_text(c): c for c in df.columns}
    if "timestamp" in cols_norm and "variable" in cols_norm and "value" in cols_norm:
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df[cols_norm["timestamp"]], errors="coerce"),
                "variable": df[cols_norm["variable"]].astype(str).map(lambda x: infer_variable(x) or normalize_text(x)),
                "value": pd.to_numeric(df[cols_norm["value"]], errors="coerce"),
            }
        )
        out = out.dropna(subset=["timestamp", "value", "variable"])
        method = "already_long"
        confidence = 0.99
    else:
        out = pd.DataFrame(columns=["timestamp", "variable", "value"])

    if out.empty and r_year is not None:
        # Pattern 2: template day rows + year columns in header row
        header = df.iloc[r_year].tolist()
        year_cols: list[tuple[int, int]] = []
        started = False
        for j, v in enumerate(header):
            y = to_float(v)
            if y is not None and 1800 <= int(y) <= 2100:
                year_cols.append((j, int(y)))
                started = True
            elif started:
                break

        date_col_idx = 0
        best_date_ratio = -1.0
        best_day_ratio = -1.0
        day_mode = False
        for c in range(min(6, df.shape[1])):
            col_vals = df.iloc[r_year + 1 :, c]
            s = pd.to_datetime(col_vals, errors="coerce")
            date_ratio = float(s.notna().mean()) if len(s) else 0.0
            day_vals = pd.to_numeric(col_vals, errors="coerce")
            day_ratio = float(day_vals.between(1, 31).mean()) if len(day_vals) else 0.0
            if date_ratio > best_date_ratio:
                best_date_ratio = date_ratio
                date_col_idx = c
                day_mode = False
            if day_ratio > best_day_ratio and day_ratio >= 0.5:
                best_day_ratio = day_ratio
                if date_ratio < 0.3:
                    date_col_idx = c
                    day_mode = True

        month_hint = infer_month(df.iat[r_year, 0] if df.shape[1] else "", file_path.name, sheet_name)
        if month_hint is None:
            month_hint = file_month

        recs = []
        for i in range(r_year + 1, len(df)):
            if day_mode:
                day_v = to_float(df.iat[i, date_col_idx])
                if day_v is None:
                    continue
                dd = int(day_v)
                if dd < 1 or dd > 31 or month_hint is None:
                    continue
                mm = month_hint
            else:
                templ = pd.to_datetime(df.iat[i, date_col_idx], errors="coerce")
                if pd.isna(templ):
                    continue
                mm = int(templ.month)
                dd = int(templ.day)

            for j, yy in year_cols:
                if j >= df.shape[1]:
                    continue
                vv = to_float(df.iat[i, j])
                if vv is None:
                    continue
                ts = safe_date(yy, mm, dd)
                if ts is None:
                    continue
                recs.append((ts, vv))
        if recs:
            var = file_var or "unknown"
            out = pd.DataFrame(recs, columns=["timestamp", "value"])
            out["variable"] = var
            method = "template_day_year_wide"
            confidence = 0.95

    if out.empty and r_year is None:
        # Pattern 1: date + hour columns (1..24), skip if a year-header pattern exists.
        first_col = df.columns[0] if len(df.columns) else None
        if first_col is not None:
            d0 = pd.to_datetime(df[first_col], errors="coerce")
            hour_cols = detect_hour_cols(df.columns[1:])
            if d0.notna().mean() >= 0.6 and len(hour_cols) >= 12 and len(hour_cols) <= 25:
                melt = df[[first_col] + hour_cols].melt(id_vars=[first_col], var_name="hour", value_name="value")
                melt["date"] = pd.to_datetime(melt[first_col], errors="coerce")
                melt["hour_i"] = melt["hour"].apply(lambda x: int(to_float(x) or 0))
                melt["timestamp"] = melt["date"] + pd.to_timedelta(np.maximum(melt["hour_i"] - 1, 0), unit="h")
                melt["value"] = pd.to_numeric(melt["value"], errors="coerce")
                var = file_var or infer_variable("hourly table", "temperature") or "temp"
                out = melt.dropna(subset=["timestamp", "value"])[["timestamp", "value"]].copy()
                out["variable"] = var
                method = "date_hour_wide"
                confidence = 0.98

    if out.empty:
        # Pattern 3: timestamp column + variable columns
        ts_col = detect_timestamp_col(df)
        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            recs = []
            for c in df.columns:
                if c == ts_col:
                    continue
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() < max(3, int(0.03 * len(vals))):
                    continue
                var = infer_variable(c, file_path.name, sheet_name) or file_var
                if var is None:
                    continue
                tmp = pd.DataFrame({"timestamp": ts, "value": vals}).dropna()
                if tmp.empty:
                    continue
                tmp["variable"] = var
                recs.append(tmp)
            if recs:
                out = pd.concat(recs, ignore_index=True)
                method = "timestamp_variable_cols"
                confidence = 0.92

    if out.empty:
        # Pattern 4: year/month/day + variable columns
        year_col = next((c for c in df.columns if normalize_text(c) in {"year", "yil"}), None)
        month_col = next((c for c in df.columns if normalize_text(c) in {"month", "ay"}), None)
        day_col = next((c for c in df.columns if normalize_text(c) in {"day", "gun"}), None)
        if year_col and month_col and day_col:
            y = pd.to_numeric(df[year_col], errors="coerce")
            m = pd.to_numeric(df[month_col], errors="coerce")
            d = pd.to_numeric(df[day_col], errors="coerce")
            base = pd.to_datetime(
                pd.DataFrame({"year": y, "month": m, "day": d}),
                errors="coerce",
            )
            recs = []
            for c in df.columns:
                if c in {year_col, month_col, day_col}:
                    continue
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().sum() < max(3, int(0.03 * len(vals))):
                    continue
                var = infer_variable(c, file_path.name, sheet_name) or file_var
                if var is None:
                    continue
                tmp = pd.DataFrame({"timestamp": base, "value": vals}).dropna()
                if tmp.empty:
                    continue
                tmp["variable"] = var
                recs.append(tmp)
            if recs:
                out = pd.concat(recs, ignore_index=True)
                method = "ymd_variable_cols"
                confidence = 0.9

    if out.empty:
        summary = {
            "file": str(file_path),
            "sheet": sheet_name,
            "status": "no_parse",
            "rows": 0,
            "method": "none",
            "variables": "",
        }
        return out, summary

    out["unit"] = out["variable"].map(VAR_UNITS).fillna("unknown")
    out["station_id"] = station_id
    out["source_file"] = str(file_path)
    out["source_sheet"] = sheet_name
    out["source_kind"] = "numeric"
    out["method"] = method
    out["confidence"] = confidence
    out["raw_payload"] = out.apply(
        lambda r: json.dumps(
            {"ts": str(r["timestamp"]), "var": str(r["variable"]), "value": float(r["value"])},
            ensure_ascii=False,
        ),
        axis=1,
    )
    out = out.dropna(subset=["timestamp", "value"])
    out = out[out["variable"].isin(["temp", "humidity", "pressure", "precip"])]
    summary = {
        "file": str(file_path),
        "sheet": sheet_name,
        "status": "ok",
        "rows": int(len(out)),
        "method": method,
        "variables": ",".join(sorted(out["variable"].unique())),
    }
    return out, summary


def parse_date_from_filename(path: Path) -> Optional[pd.Timestamp]:
    m = re.search(r"(19|20)\d{2}[_-](\d{2})[_-](\d{2})", path.name)
    if not m:
        return None
    y = int(path.name[m.start() : m.start() + 4])
    mm = int(m.group(2))
    dd = int(m.group(3))
    return safe_date(y, mm, dd)


def load_graph_traces(graph_dir: Path | None, station_id: str) -> pd.DataFrame:
    if graph_dir is None or not graph_dir.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(graph_dir.rglob("*.csv")):
        name_n = normalize_text(p.name)
        if not name_n.endswith("_trace.csv"):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        ts = pd.to_datetime(df.get("timestamp", pd.Series(dtype=object)), errors="coerce")
        elapsed = pd.to_numeric(df.get("elapsed_hour"), errors="coerce")
        if ts.isna().all() and elapsed.notna().any():
            base = parse_date_from_filename(p)
            if base is not None:
                ts = base + pd.to_timedelta(8, unit="h") + pd.to_timedelta(elapsed.fillna(0), unit="h")

        # detect value column
        value_col = None
        for c in df.columns:
            if infer_variable(c, p.name):
                value_col = c
                break
        if value_col is None:
            value_col = "value" if "value" in df.columns else None
        if value_col is None:
            if "humidity_pct" in df.columns:
                value_col = "humidity_pct"
            else:
                continue

        var = infer_variable(value_col, p.name) or "humidity"
        val = pd.to_numeric(df[value_col], errors="coerce")
        out = pd.DataFrame(
            {
                "timestamp": ts,
                "variable": var,
                "value": val,
                "unit": VAR_UNITS.get(var, "unknown"),
                "station_id": station_id,
                "source_file": str(p),
                "source_sheet": "graph_trace",
                "source_kind": "graph_paper",
                "method": "trace_csv",
                "confidence": 0.78,
            }
        )
        out = out.dropna(subset=["timestamp", "value"])
        if out.empty:
            continue
        out["raw_payload"] = out.apply(
            lambda r: json.dumps({"ts": str(r["timestamp"]), "var": str(r["variable"]), "value": float(r["value"])}, ensure_ascii=False),
            axis=1,
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def qc_flag(variable: str, value: float) -> str:
    if pd.isna(value):
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


def aggregate_series(raw: pd.Series, variable: str, freq_key: str, freq_code: str) -> pd.DataFrame:
    raw = raw.sort_index()
    agg_method = "sum" if variable == "precip" and freq_key in {"daily", "monthly", "yearly"} else "mean"
    if agg_method == "sum":
        rs = raw.resample(freq_code).sum(min_count=1)
    else:
        rs = raw.resample(freq_code).mean()
    observed = rs.notna()
    filled = rs.copy()
    if variable == "precip":
        filled = filled.fillna(0.0)
    else:
        filled = filled.interpolate("time").ffill().bfill()
    if filled.isna().all():
        filled = pd.Series(np.zeros(len(filled)), index=filled.index)
    out = pd.DataFrame(
        {
            "timestamp": filled.index,
            "value": filled.values.astype(float),
            "observed": observed.values,
            "observed_ratio": float(observed.mean()) if len(observed) else 0.0,
        }
    )
    out["variable"] = variable
    out["frequency"] = freq_key
    out["unit"] = VAR_UNITS.get(variable, "unknown")
    return out


def seasonal_trend_forecast(series: pd.Series, horizon: int, seasonal_period: int) -> pd.DataFrame:
    y = np.asarray(series.values, dtype=float)
    n = len(y)
    if n == 0:
        return pd.DataFrame(columns=["yhat", "low", "high"])
    if n == 1:
        pred = np.repeat(y[0], horizon)
        return pd.DataFrame({"yhat": pred, "low": pred, "high": pred})
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    trend = intercept + slope * t
    p = max(2, min(seasonal_period, n // 2 if n >= 4 else 2))
    resid = y - trend
    season = np.zeros(p, dtype=float)
    for i in range(p):
        vals = resid[i::p]
        season[i] = float(np.nanmean(vals)) if len(vals) else 0.0
    fit = trend + season[(t.astype(int) % p)]
    sigma = float(np.nanstd(y - fit))
    if not np.isfinite(sigma):
        sigma = 0.0
    future_t = np.arange(n, n + horizon, dtype=float)
    yhat = intercept + slope * future_t + season[(future_t.astype(int) % p)]
    low = yhat - 1.96 * sigma
    high = yhat + 1.96 * sigma
    return pd.DataFrame({"yhat": yhat, "low": low, "high": high})


def apply_bounds(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    out = df.copy()
    if variable == "humidity":
        for c in ["yhat", "low", "high"]:
            out[c] = out[c].clip(0, 100)
    elif variable in {"precip", "pressure"}:
        for c in ["yhat", "low", "high"]:
            out[c] = out[c].clip(lower=0)
    return out


def make_forecast_dataset(hist: pd.DataFrame, freq_code: str, horizon: int, seasonal_period: int, variable: str) -> pd.DataFrame:
    hist = hist.sort_values("timestamp").reset_index(drop=True)
    fc = seasonal_trend_forecast(hist["value"], horizon=horizon, seasonal_period=seasonal_period)
    if fc.empty:
        return pd.DataFrame(columns=["timestamp", "yhat", "low", "high", "is_forecast"])
    last_ts = pd.Timestamp(hist["timestamp"].iloc[-1])
    future_idx = pd.date_range(last_ts + to_offset(freq_code), periods=horizon, freq=freq_code)
    fc_out = pd.DataFrame(
        {
            "timestamp": future_idx,
            "yhat": fc["yhat"].values,
            "low": fc["low"].values,
            "high": fc["high"].values,
            "is_forecast": True,
        }
    )
    hist_out = pd.DataFrame(
        {
            "timestamp": hist["timestamp"],
            "yhat": hist["value"],
            "low": np.nan,
            "high": np.nan,
            "is_forecast": False,
        }
    )
    out = pd.concat([hist_out, fc_out], ignore_index=True)
    out = apply_bounds(out, variable)
    return out


def plot_forecast(df: pd.DataFrame, variable: str, freq_key: str, out_path: Path) -> None:
    hist = df[df["is_forecast"] == False]  # noqa: E712
    fc = df[df["is_forecast"] == True]  # noqa: E712
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist["timestamp"], hist["yhat"], label="Gercek/Islenmis", linewidth=1.1)
    if not fc.empty:
        ax.plot(fc["timestamp"], fc["yhat"], label="Tahmin", color="tab:orange", linewidth=1.6)
        ax.fill_between(fc["timestamp"], fc["low"], fc["high"], color="tab:orange", alpha=0.2, label="Guven bandi")
    ax.set_title(f"{FREQ_CONFIG[freq_key]['title']} Tahmin - {variable}")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Deger")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close(fig)


def save_df(df: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    df.to_parquet(base.with_suffix(".parquet"), index=False)


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    datasets_dir = out_dir / "datasets"
    forecasts_dir = out_dir / "forecasts"
    charts_dir = out_dir / "charts"
    for d in [out_dir, datasets_dir, forecasts_dir, charts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    parse_rows = []
    obs_frames = []

    files = list_files(args.input_dir)
    for p in files:
        sheets = read_tabular_file(p)
        if not sheets:
            parse_rows.append(
                {"file": str(p), "sheet": "", "status": "read_fail", "rows": 0, "method": "", "variables": ""}
            )
            continue
        for sname, df in sheets:
            out, summary = extract_from_df(df, p, sname, args.station_id)
            parse_rows.append(summary)
            if not out.empty:
                obs_frames.append(out)

    graph_df = load_graph_traces(args.graph_dir, station_id=args.station_id)
    if not graph_df.empty:
        obs_frames.append(graph_df)

    if not obs_frames:
        raise SystemExit("No observations parsed from inputs.")

    obs = pd.concat(obs_frames, ignore_index=True)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.dropna(subset=["timestamp", "value", "variable"])
    obs["qc_flag"] = [qc_flag(v, x) for v, x in zip(obs["variable"], obs["value"])]
    obs["is_missing"] = False
    obs["year"] = obs["timestamp"].dt.year
    obs["month"] = obs["timestamp"].dt.month
    obs["day"] = obs["timestamp"].dt.day
    obs["hour"] = obs["timestamp"].dt.hour
    obs = obs.sort_values(["timestamp", "variable"]).reset_index(drop=True)

    save_df(obs, out_dir / "observations_universal")
    pd.DataFrame(parse_rows).to_csv(out_dir / "parse_report.csv", index=False)

    src_summary = (
        obs.groupby(["variable", "source_kind"])
        .agg(rows=("value", "size"), min_ts=("timestamp", "min"), max_ts=("timestamp", "max"))
        .reset_index()
    )
    src_summary.to_csv(out_dir / "source_summary.csv", index=False)

    enabled_freqs = [x.strip() for x in args.freqs.split(",") if x.strip() in FREQ_CONFIG]
    if not enabled_freqs:
        raise SystemExit("No valid frequencies requested.")

    index_rows = []
    for variable in sorted(obs["variable"].unique()):
        sub = obs[(obs["variable"] == variable) & (obs["qc_flag"] == "ok")]
        if sub.empty:
            continue
        raw = sub.groupby("timestamp")["value"].mean().sort_index()
        for freq_key in enabled_freqs:
            cfg = FREQ_CONFIG[freq_key]
            hist = aggregate_series(raw, variable, freq_key=freq_key, freq_code=cfg["freq"])
            if hist.empty:
                continue
            hist_base = datasets_dir / f"{variable}_{freq_key}_historical"
            save_df(hist, hist_base)

            fc = make_forecast_dataset(
                hist=hist[["timestamp", "value"]],
                freq_code=cfg["freq"],
                horizon=cfg["horizon"],
                seasonal_period=cfg["seasonal_period"],
                variable=variable,
            )
            fc["variable"] = variable
            fc["frequency"] = freq_key
            fc_base = forecasts_dir / f"{variable}_{freq_key}_forecast"
            save_df(fc, fc_base)
            chart_p = charts_dir / f"{variable}_{freq_key}_forecast.png"
            plot_forecast(fc, variable=variable, freq_key=freq_key, out_path=chart_p)

            index_rows.append(
                {
                    "variable": variable,
                    "frequency": freq_key,
                    "historical_rows": len(hist),
                    "observed_ratio": float(hist["observed_ratio"].iloc[0]) if len(hist) else np.nan,
                    "forecast_rows": int((fc["is_forecast"] == True).sum()),  # noqa: E712
                    "historical_start": hist["timestamp"].min(),
                    "historical_end": hist["timestamp"].max(),
                    "forecast_end": fc["timestamp"].max(),
                    "historical_file_csv": str(hist_base.with_suffix(".csv")),
                    "forecast_file_csv": str(fc_base.with_suffix(".csv")),
                    "chart_file": str(chart_p),
                }
            )

    idx = pd.DataFrame(index_rows).sort_values(["variable", "frequency"])
    idx.to_csv(out_dir / "forecast_index.csv", index=False)
    idx.to_parquet(out_dir / "forecast_index.parquet", index=False)

    print("Pipeline completed.")
    print(f"- observations: {out_dir / 'observations_universal.parquet'}")
    print(f"- parse report: {out_dir / 'parse_report.csv'}")
    print(f"- source summary: {out_dir / 'source_summary.csv'}")
    print(f"- forecast index: {out_dir / 'forecast_index.csv'}")
    if not idx.empty:
        print(idx[["variable", "frequency", "historical_rows", "forecast_rows"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
