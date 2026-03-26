#!/usr/bin/env python3
"""Best-effort meta ensemble for climate forecasts.

Design goals:
- Orchestrate strongest base models (quant, strong, prophet_ultra, walkforward)
- Build per-variable weighted ensemble using CV quality + stability + bias penalties
- Apply consistent climate-scenario adjustment to all model members
- Reduce long-horizon recursive drift via mild monthly mean reversion
- Emit production-style outputs (forecast/index/reports/charts + annual comparisons)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climate_scenario_adjustment import (
    ClimateAdjustmentConfig,
    climate_delta_series,
    from_args as climate_from_args,
    with_series_baseline,
)

# Avoid cache warnings on restricted environments.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "best_meta_cache"
_MPL_CACHE = _CACHE_ROOT / "mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

UNIT_MAP = {"humidity": "%", "temp": "C", "pressure": "hPa", "precip": "mm"}

TR_CHARMAP = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ş": "s",
        "Ş": "s",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    }
)

ALIASES = {
    "nem": "humidity",
    "humidity": "humidity",
    "relative_humidity": "humidity",
    "rh": "humidity",
    "sicaklik": "temp",
    "sıcaklık": "temp",
    "temperature": "temp",
    "temp": "temp",
    "basinc": "pressure",
    "basınç": "pressure",
    "pressure": "pressure",
    "pres": "pressure",
    "yagis": "precip",
    "yağış": "precip",
    "precip": "precip",
    "precipitation": "precip",
    "rain": "precip",
    "rainfall": "precip",
    "prcp": "precip",
}


@dataclass
class ModelScore:
    model_key: str
    variable: str
    frequency: str
    forecast_csv: str
    rmse: float
    bias: float
    rmse_std: float
    has_cv: bool
    source: str
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best climate meta ensemble (quant+strong+prophet_ultra+walkforward)")
    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/best_meta_ensemble"),
    )
    p.add_argument("--variables", type=str, default="temp,humidity")
    p.add_argument("--target-year", type=int, default=2035)
    p.add_argument("--walkforward-start-year", type=int, default=2026)
    p.add_argument(
        "--forecast-start-year",
        type=int,
        default=None,
        help="Meta forecast alt siniri (yil). Bos ise walkforward-start-year kullanilir.",
    )
    p.add_argument("--input-kind", type=str, default="auto", choices=["auto", "long", "single"])
    p.add_argument("--timestamp-col", type=str, default="timestamp")
    p.add_argument("--value-col", type=str, default="value")
    p.add_argument("--variable-col", type=str, default="variable")
    p.add_argument("--qc-col", type=str, default="qc_flag")
    p.add_argument("--qc-ok-value", type=str, default="ok")
    p.add_argument("--single-variable", type=str, default="target")
    p.add_argument("--max-model-weight", type=float, default=0.65)
    p.add_argument("--min-model-weight", type=float, default=0.05)
    p.add_argument("--interval-z", type=float, default=1.64)
    p.add_argument(
        "--auto-select-combination",
        type=str,
        default="true",
        help="true/false: model alt-kumesi ve agirliklarini rolling-CV ile otomatik sec.",
    )
    p.add_argument(
        "--max-combo-size",
        type=int,
        default=3,
        help="Otomatik secimde izin verilen maksimum model sayisi.",
    )
    p.add_argument(
        "--combo-complexity-penalty",
        type=float,
        default=0.025,
        help="Fazla model kullanimi ceza katsayisi (0 kapali).",
    )
    p.add_argument(
        "--force-combine-count",
        type=int,
        default=0,
        help="0 ise kapali. >1 ise en iyi N modeli zorunlu birlikte birlestirir (10-15 icin kullan).",
    )
    p.add_argument(
        "--auto-tune-force-combine",
        type=str,
        default="true",
        help="true/false. true ise her degisken icin force_combine_count adaylarini deneyip en stabil sonucu secer.",
    )
    p.add_argument(
        "--force-combine-candidates",
        type=str,
        default="8,10,12,15",
        help="Virgulle ayrilmis aday model sayilari (or: 8,10,12,15).",
    )
    p.add_argument(
        "--auto-tune-continuity-alpha",
        type=str,
        default="true",
        help="true/false. true ise force_combine taramasinda continuity alpha da birlikte optimize edilir.",
    )
    p.add_argument(
        "--continuity-alpha-candidates",
        type=str,
        default="",
        help="Virgulle ayrilmis continuity alpha adaylari (or: 0.04,0.08,0.12). Bos ise degisken-bazli otomatik kullanilir.",
    )
    p.add_argument("--run-base-models", type=str, default="true", help="true/false")
    p.add_argument("--reuse-existing", type=str, default="true", help="true/false")
    p.add_argument("--base-dir-name", type=str, default="base_models")
    p.add_argument("--climate-scenario", type=str, default="ssp245")
    p.add_argument("--climate-baseline-year", type=float, default=float("nan"))
    p.add_argument("--climate-temp-rate", type=float, default=float("nan"))
    p.add_argument("--humidity-per-temp-c", type=float, default=-2.0)
    p.add_argument("--climate-adjustment-method", type=str, default="pathway")
    p.add_argument("--disable-climate-adjustment", action="store_true")
    p.add_argument(
        "--enable-temp-proxy",
        type=str,
        default="false",
        help="true/false. true ise nemden turetilen sicaklik proxy noktalarini ekler.",
    )
    p.add_argument(
        "--min-temp-warming-rate-c-per-year",
        type=float,
        default=0.0,
        help="Sicaklik forecast yillik egimi icin alt sinir. 0 ile kapali (zorunlu trend yok).",
    )
    return p.parse_args()


def to_bool(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def parse_int_csv(text: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for tok in str(text).split(","):
        t = str(tok).strip()
        if not t:
            continue
        try:
            v = int(float(t))
        except Exception:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(int(v))
    return out


def parse_float_csv(text: Any) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for tok in str(text).split(","):
        t = str(tok).strip()
        if not t:
            continue
        try:
            v = float(t)
        except Exception:
            continue
        v = float(np.clip(v, 0.0, 0.9))
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def normalize_token(text: Any) -> str:
    s = str(text).strip().lower().translate(TR_CHARMAP)
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonical_variable_name(text: Any) -> str:
    t = normalize_token(text)
    if t in ALIASES:
        return ALIASES[t]
    if any(k in t for k in ["humid", "nem", "rh"]):
        return "humidity"
    if any(k in t for k in ["temp", "sicak", "sicaklik", "temperature", "t2m"]):
        return "temp"
    if any(k in t for k in ["press", "basinc", "hpa", "mbar"]):
        return "pressure"
    if any(k in t for k in ["precip", "rain", "yagis", "prcp"]):
        return "precip"
    return t if t else "target"


def infer_unit(variable: str) -> str:
    return UNIT_MAP.get(canonical_variable_name(variable), "unknown")


def is_humidity(variable: str) -> bool:
    return canonical_variable_name(variable) == "humidity"


def is_precip(variable: str) -> bool:
    return canonical_variable_name(variable) == "precip"


def is_pressure(variable: str) -> bool:
    return canonical_variable_name(variable) == "pressure"


def apply_bounds(arr: np.ndarray, variable: str) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if is_humidity(variable):
        return np.clip(x, 0, 100)
    if canonical_variable_name(variable) == "temp":
        # Physical guardrail for near-surface air temperature forecasts.
        return np.clip(x, -50.0, 60.0)
    if is_precip(variable) or is_pressure(variable):
        return np.clip(x, 0, None)
    return x


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf in {".xlsx", ".xls", ".ods"}:
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported extension: {path.suffix}")


def pick_existing_column(raw: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str | None:
    if preferred in raw.columns:
        return preferred
    for c in fallbacks:
        if c in raw.columns:
            return c
    return None


def normalize_observations(raw: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    ts_col = pick_existing_column(raw, args.timestamp_col, ["timestamp", "ds", "date", "datetime", "time", "tarih"])
    val_col = pick_existing_column(raw, args.value_col, ["value", "y", "target", "measurement"])
    var_col = pick_existing_column(raw, args.variable_col, ["variable", "metric", "param", "sensor", "name"])
    qc_col = pick_existing_column(raw, args.qc_col, ["qc_flag", "qc", "quality", "flag"])
    if ts_col is None or val_col is None:
        raise SystemExit("Cannot detect timestamp/value columns")

    kind = args.input_kind
    if kind == "auto":
        kind = "long" if var_col is not None else "single"

    if kind == "long":
        if var_col is None:
            raise SystemExit("input-kind=long requires variable column")
        out = pd.DataFrame({"timestamp": raw[ts_col], "variable": raw[var_col], "value": raw[val_col]})
    else:
        out = pd.DataFrame({"timestamp": raw[ts_col], "variable": args.single_variable, "value": raw[val_col]})

    if qc_col is not None:
        out["qc_flag"] = raw[qc_col]
    else:
        out["qc_flag"] = args.qc_ok_value

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["variable"] = out["variable"].astype(str).map(canonical_variable_name)
    out["qc_flag"] = out["qc_flag"].astype(str)
    out = out.dropna(subset=["timestamp", "value", "variable"]).sort_values("timestamp").reset_index(drop=True)
    return out, kind


def requested_variables(obs: pd.DataFrame, variables_arg: str) -> list[str]:
    avail = sorted(obs["variable"].dropna().unique().tolist())
    if not variables_arg or variables_arg.strip() in {"*", "all", "ALL"}:
        return avail
    req = [canonical_variable_name(v.strip()) for v in str(variables_arg).split(",") if v.strip()]
    req = [v for v in req if v in avail]
    return sorted(set(req))


def choose_monthly_observed_series(obs: pd.DataFrame, variable: str, ok_value: str) -> pd.Series:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    ok = sub["qc_flag"].astype(str).str.lower().eq(str(ok_value).lower())
    if ok.any():
        sub = sub[ok]
    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return pd.Series(dtype=float)
    raw = sub.groupby("timestamp")["value"].mean().sort_index()
    if is_precip(variable):
        # Keep missing months as NaN to avoid fabricating long dry spells.
        s = raw.resample("MS").sum(min_count=1)
    else:
        # Fill only short interior gaps (<=2 months). Avoid ffill/bfill over long spans,
        # which can create unrealistic multi-year pseudo-history on sparse sources.
        s = raw.resample("MS").mean().interpolate("time", limit=2, limit_area="inside")
    return s.astype(float)


def monthly_from_obs_no_fill(obs: pd.DataFrame, variable: str, ok_value: str) -> pd.Series:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    ok = sub["qc_flag"].astype(str).str.lower().eq(str(ok_value).lower())
    if ok.any():
        sub = sub[ok]
    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return pd.Series(dtype=float)
    raw = sub.groupby("timestamp")["value"].mean().sort_index()
    if is_precip(variable):
        return raw.resample("MS").sum(min_count=1).astype(float)
    return raw.resample("MS").mean().astype(float)


def augment_temp_from_humidity_proxy(obs: pd.DataFrame, ok_value: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": False,
        "reason": "not_applicable",
        "added_points": 0,
        "temp_months_before": 0,
        "temp_months_after": 0,
        "humidity_months": 0,
        "overlap_months": 0,
        "train_rmse_c": None,
    }
    if obs.empty:
        stats["reason"] = "empty_observations"
        return obs, stats
    vars_present = set(obs["variable"].astype(str).unique().tolist())
    if "temp" not in vars_present or "humidity" not in vars_present:
        stats["reason"] = "missing_temp_or_humidity"
        return obs, stats

    temp_m = monthly_from_obs_no_fill(obs, variable="temp", ok_value=ok_value)
    hum_m = monthly_from_obs_no_fill(obs, variable="humidity", ok_value=ok_value)
    if hum_m.empty:
        stats["reason"] = "empty_humidity"
        return obs, stats

    cov_temp_before = monthly_history_coverage(temp_m)
    stats["temp_months_before"] = int(cov_temp_before["months_with_data"])
    stats["humidity_months"] = int(monthly_history_coverage(hum_m)["months_with_data"])

    # Only augment when temp history is clearly sparse but humidity is rich enough.
    if stats["temp_months_before"] >= 120 or stats["humidity_months"] < 60:
        stats["reason"] = "coverage_not_sparse_enough_or_humidity_insufficient"
        return obs, stats

    idx = temp_m.index.union(hum_m.index).sort_values()
    temp_a = temp_m.reindex(idx)
    hum_a = hum_m.reindex(idx)
    overlap = temp_a.notna() & hum_a.notna()
    need = temp_a.isna() & hum_a.notna()
    stats["overlap_months"] = int(overlap.sum())

    if overlap.sum() < 8 or need.sum() < 6:
        stats["reason"] = "insufficient_overlap_or_missing_months"
        return obs, stats

    train_idx = idx[overlap.values]
    pred_idx = idx[need.values]
    y = temp_a.loc[train_idx].values.astype(float)
    h_train = hum_a.loc[train_idx].values.astype(float)
    h_pred = hum_a.loc[pred_idx].values.astype(float)

    year_mean = float(np.mean(train_idx.year.values.astype(float)))

    def design(ix: pd.DatetimeIndex, h: np.ndarray, year_center: float) -> np.ndarray:
        month = ix.month.values.astype(float)
        year = ix.year.values.astype(float)
        sin_m = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        cos_m = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        year_c = year - year_center
        return np.column_stack([np.ones(len(ix)), h, sin_m, cos_m, year_c])

    X = design(train_idx, h_train, year_center=year_mean)
    Xp = design(pred_idx, h_pred, year_center=year_mean)

    # Lightweight ridge for stability on small overlap sample.
    lam = 0.8
    reg = np.eye(X.shape[1], dtype=float) * lam
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    y_fit = X @ beta
    rmse = float(np.sqrt(np.mean((y - y_fit) ** 2)))
    y_hat = Xp @ beta
    y_hat = np.clip(y_hat, -25.0, 45.0)

    h_lo = float(np.nanmin(h_train))
    h_hi = float(np.nanmax(h_train))
    extra = np.maximum(h_lo - h_pred, 0.0) + np.maximum(h_pred - h_hi, 0.0)
    base_conf = float(np.clip(1.0 - (rmse / 14.0), 0.30, 0.82))
    conf = np.clip(base_conf * np.exp(-extra / 18.0), 0.22, 0.82)

    rows = []
    for ts, v, c in zip(pred_idx, y_hat, conf):
        if not np.isfinite(v):
            continue
        rows.append(
            {
                "timestamp": pd.Timestamp(ts) + pd.Timedelta(hours=12),
                "variable": "temp",
                "value": float(v),
                "qc_flag": str(ok_value),
                "source_kind": "derived_proxy",
                "method": "temp_proxy_from_humidity_monthly_ridge",
                "confidence": float(c),
            }
        )
    if not rows:
        stats["reason"] = "no_proxy_rows_generated"
        return obs, stats

    add = pd.DataFrame(rows)
    merged = pd.concat([obs.copy(), add], ignore_index=True, sort=False)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged = merged.dropna(subset=["timestamp", "value"]).sort_values(["timestamp", "variable"]).reset_index(drop=True)

    temp_after = monthly_from_obs_no_fill(merged, variable="temp", ok_value=ok_value)
    cov_after = monthly_history_coverage(temp_after)

    stats.update(
        {
            "enabled": True,
            "reason": "ok",
            "added_points": int(len(rows)),
            "temp_months_after": int(cov_after["months_with_data"]),
            "train_rmse_c": rmse,
            "train_humidity_min": h_lo,
            "train_humidity_max": h_hi,
            "train_overlap_start": str(pd.Timestamp(train_idx.min()).date()),
            "train_overlap_end": str(pd.Timestamp(train_idx.max()).date()),
        }
    )
    return merged, stats


def monthly_history_coverage(obs_monthly: pd.Series, reference_end: pd.Timestamp | None = None) -> dict[str, Any]:
    s = pd.to_numeric(obs_monthly, errors="coerce")
    valid = s.dropna().sort_index()
    if valid.empty:
        return {
            "months_with_data": 0,
            "years_with_data": 0,
            "span_months": 0,
            "span_months_raw": 0,
            "span_months_effective": 0,
            "coverage_ratio": 0.0,
            "month_of_year_coverage": 0,
            "max_internal_gap_months": 0,
            "recency_gap_months": 0,
            "recency_gap_years": 0.0,
            "history_start": None,
            "history_end": None,
            "history_reference_end": str(pd.Timestamp(reference_end).date()) if reference_end is not None else None,
            "quality_flag": "low",
            "sparse_history": True,
        }

    idx = pd.to_datetime(valid.index)
    start = pd.Timestamp(idx.min())
    end = pd.Timestamp(idx.max())
    span_raw = int((end.year - start.year) * 12 + (end.month - start.month) + 1)
    span_raw = max(span_raw, 1)

    ref_end = pd.Timestamp(reference_end) if reference_end is not None else end
    if ref_end < end:
        ref_end = end
    span_eff = int((ref_end.year - start.year) * 12 + (ref_end.month - start.month) + 1)
    span_eff = max(span_eff, 1)

    recency_gap = int(max((ref_end.year - end.year) * 12 + (ref_end.month - end.month), 0))
    recency_gap_years = float(recency_gap / 12.0)

    months_with_data = int(len(valid))
    years_with_data = int(pd.Index(idx.year).nunique())
    month_of_year_coverage = int(pd.Index(idx.month).nunique())
    coverage_ratio = float(months_with_data / span_eff)

    month_ids = (idx.year * 12 + idx.month).astype(int)
    if len(month_ids) <= 1:
        max_gap = 0
    else:
        gaps = np.diff(month_ids)
        max_gap = int(np.max(np.maximum(gaps - 1, 0)))

    if months_with_data < 36 or years_with_data < 5 or coverage_ratio < 0.40:
        quality = "low"
    elif months_with_data < 84 or years_with_data < 10 or coverage_ratio < 0.70:
        quality = "medium"
    else:
        quality = "high"
    if recency_gap >= 60 and quality != "low":
        quality = "medium"
    if recency_gap >= 120:
        quality = "low"

    return {
        "months_with_data": months_with_data,
        "years_with_data": years_with_data,
        "span_months": span_eff,
        "span_months_raw": span_raw,
        "span_months_effective": span_eff,
        "coverage_ratio": coverage_ratio,
        "month_of_year_coverage": month_of_year_coverage,
        "max_internal_gap_months": max_gap,
        "recency_gap_months": recency_gap,
        "recency_gap_years": recency_gap_years,
        "history_start": str(start.date()),
        "history_end": str(end.date()),
        "history_reference_end": str(ref_end.date()),
        "quality_flag": quality,
        "sparse_history": bool(quality == "low"),
    }


def smooth_month_climatology(obs_valid: pd.Series, variable: str) -> dict[int, float]:
    s = pd.to_numeric(obs_valid, errors="coerce").dropna().sort_index()
    if s.empty:
        return {}
    mm = s.groupby(s.index.month).mean()
    cc = s.groupby(s.index.month).count()
    if len(mm) < 4:
        return {int(k): float(v) for k, v in mm.items()}

    months = mm.index.values.astype(float)
    y = mm.values.astype(float)
    w = np.sqrt(np.maximum(cc.reindex(mm.index).values.astype(float), 1.0))
    ang = 2.0 * np.pi * (months - 1.0) / 12.0
    X = np.column_stack(
        [
            np.ones(len(months), dtype=float),
            np.sin(ang),
            np.cos(ang),
            np.sin(2.0 * ang),
            np.cos(2.0 * ang),
        ]
    )
    Xw = X * w[:, None]
    yw = y * w
    reg = np.eye(X.shape[1], dtype=float) * 0.4
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]

    all_m = np.arange(1, 13, dtype=float)
    a2 = 2.0 * np.pi * (all_m - 1.0) / 12.0
    Xa = np.column_stack(
        [
            np.ones(len(all_m), dtype=float),
            np.sin(a2),
            np.cos(a2),
            np.sin(2.0 * a2),
            np.cos(2.0 * a2),
        ]
    )
    pred = Xa @ beta
    out: dict[int, float] = {}
    for i, m in enumerate(all_m.astype(int).tolist()):
        p = float(pred[i])
        if m in mm.index:
            raw = float(mm.loc[m])
            cnt = float(cc.loc[m])
            raw_w = float(np.clip(cnt / (cnt + 4.0), 0.20, 0.85))
            val = raw_w * raw + (1.0 - raw_w) * p
        else:
            val = p
        if is_humidity(variable):
            val = float(np.clip(val, 0.0, 100.0))
        elif is_precip(variable) or is_pressure(variable):
            val = float(max(0.0, val))
        out[int(m)] = val
    return out


def apply_consistency_regularization(
    blend: pd.DataFrame,
    obs_valid: pd.Series,
    variable: str,
    coverage: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": False,
        "reason": "not_applied",
        "shrink_to_climatology": 0.0,
        "jump_cap": 0.0,
        "mean_abs_shift": 0.0,
        "max_abs_shift": 0.0,
    }
    if blend is None or blend.empty:
        stats["reason"] = "empty_blend"
        return blend, stats

    month_clim = smooth_month_climatology(obs_valid, variable=variable)
    if len(month_clim) < 6:
        stats["reason"] = "insufficient_month_climatology"
        return blend, stats

    out = blend.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    y = pd.to_numeric(out["yhat"], errors="coerce").values.astype(float)
    lo = pd.to_numeric(out["yhat_lower"], errors="coerce").values.astype(float)
    hi = pd.to_numeric(out["yhat_upper"], errors="coerce").values.astype(float)

    recency_gap_years = float(coverage.get("recency_gap_years", 0.0))
    cov_ratio = float(coverage.get("coverage_ratio", 1.0))
    months_with_data = int(coverage.get("months_with_data", 0))
    max_internal_gap = float(coverage.get("max_internal_gap_months", 0))

    shrink = 0.10 + 0.018 * max(recency_gap_years, 0.0) + 0.30 * max(0.0, 1.0 - cov_ratio) + 0.01 * max_internal_gap
    if months_with_data < 24:
        shrink = max(shrink, 0.58)
    elif months_with_data < 60:
        shrink = max(shrink, 0.40)

    if is_precip(variable):
        shrink = float(np.clip(shrink, 0.45, 0.85))
    elif is_humidity(variable):
        shrink = float(np.clip(shrink, 0.25, 0.72))
    else:
        shrink = float(np.clip(shrink, 0.20, 0.78))

    shift_hist = np.zeros(len(out), dtype=float)
    for i, ts in enumerate(out["ds"].tolist()):
        if pd.isna(ts):
            continue
        m = int(pd.Timestamp(ts).month)
        anchor = float(month_clim.get(m, y[i]))
        prev = float(y[i])
        now = (1.0 - shrink) * prev + shrink * anchor
        shift = now - prev
        y[i] = now
        lo[i] = lo[i] + shift if np.isfinite(lo[i]) else lo[i]
        hi[i] = hi[i] + shift if np.isfinite(hi[i]) else hi[i]
        shift_hist[i] = shift

    hist_diff = pd.to_numeric(obs_valid, errors="coerce").dropna().diff().abs().dropna()
    if len(hist_diff) >= 4:
        q90 = float(hist_diff.quantile(0.90))
        q95 = float(hist_diff.quantile(0.95))
    else:
        q90 = np.nan
        q95 = np.nan
    jump_cap = q95 * 1.20 if np.isfinite(q95) else np.nan
    if variable == "temp":
        jump_cap = float(np.clip(jump_cap if np.isfinite(jump_cap) else 3.5, 1.5, 7.0))
    elif variable == "humidity":
        jump_cap = float(np.clip(jump_cap if np.isfinite(jump_cap) else 5.0, 2.0, 12.0))
    elif variable == "precip":
        base_p = q90 * 1.05 if np.isfinite(q90) else (jump_cap if np.isfinite(jump_cap) else 60.0)
        jump_cap = float(np.clip(base_p, 20.0, 90.0))
    else:
        jump_cap = float(np.clip(jump_cap if np.isfinite(jump_cap) else 8.0, 2.0, 40.0))

    for i in range(1, len(y)):
        if not np.isfinite(y[i - 1]) or not np.isfinite(y[i]):
            continue
        low = y[i - 1] - jump_cap
        high = y[i - 1] + jump_cap
        new_y = float(np.clip(y[i], low, high))
        if new_y != y[i]:
            shift = new_y - y[i]
            y[i] = new_y
            lo[i] = lo[i] + shift if np.isfinite(lo[i]) else lo[i]
            hi[i] = hi[i] + shift if np.isfinite(hi[i]) else hi[i]
            shift_hist[i] += shift

    y = apply_bounds(y, variable)
    lo = apply_bounds(lo, variable)
    hi = apply_bounds(hi, variable)
    lo = np.minimum(lo, y)
    hi = np.maximum(hi, y)
    out["yhat"] = y
    out["yhat_lower"] = lo
    out["yhat_upper"] = hi

    stats.update(
        {
            "enabled": True,
            "reason": "ok",
            "shrink_to_climatology": float(shrink),
            "jump_cap": float(jump_cap),
            "mean_abs_shift": float(np.mean(np.abs(shift_hist))) if len(shift_hist) else 0.0,
            "max_abs_shift": float(np.max(np.abs(shift_hist))) if len(shift_hist) else 0.0,
        }
    )
    return out, stats


def continuity_metrics(ds: pd.Series, y: np.ndarray) -> dict[str, float]:
    yy = np.asarray(y, dtype=float)
    if yy.size < 3:
        return {
            "jump_q95": float("inf"),
            "jump_max": float("inf"),
            "ann_jump_q95": float("inf"),
            "ann_jump_max": float("inf"),
            "amp_p90_p10": 0.0,
            "mean": float(np.nanmean(yy)) if yy.size else 0.0,
        }
    dd = np.abs(np.diff(yy))
    jump_q95 = float(np.quantile(dd, 0.95)) if dd.size else 0.0
    jump_max = float(np.max(dd)) if dd.size else 0.0
    ann_jump_q95 = 0.0
    ann_jump_max = 0.0
    try:
        td = pd.to_datetime(ds, errors="coerce")
        ann = pd.DataFrame({"year": td.dt.year, "y": yy}).dropna(subset=["year", "y"])
        ann = ann.groupby("year", as_index=False)["y"].mean().dropna()
        ad = np.abs(np.diff(ann["y"].values.astype(float)))
        ann_jump_q95 = float(np.quantile(ad, 0.95)) if ad.size else 0.0
        ann_jump_max = float(np.max(ad)) if ad.size else 0.0
    except Exception:
        ann_jump_q95 = 0.0
        ann_jump_max = 0.0
    amp = float(np.quantile(yy, 0.90) - np.quantile(yy, 0.10)) if yy.size else 0.0
    return {
        "jump_q95": float(jump_q95),
        "jump_max": float(jump_max),
        "ann_jump_q95": float(ann_jump_q95),
        "ann_jump_max": float(ann_jump_max),
        "amp_p90_p10": float(max(0.0, amp)),
        "mean": float(np.nanmean(yy)) if yy.size else 0.0,
    }


def median3_smooth(y: np.ndarray, alpha: float, passes: int = 2) -> np.ndarray:
    yy = np.asarray(y, dtype=float).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    if yy.size < 3 or a <= 1e-9:
        return yy
    pcount = int(max(1, passes))
    for _ in range(pcount):
        prev = yy.copy()
        for i in range(1, len(yy) - 1):
            c0, c1, c2 = prev[i - 1], prev[i], prev[i + 1]
            if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(c2)):
                continue
            med = float(np.median([c0, c1, c2]))
            yy[i] = (1.0 - a) * c1 + a * med
    return yy


def continuity_alpha_candidates(variable: str, coverage: dict[str, Any]) -> list[float]:
    var = canonical_variable_name(variable)
    if var == "temp":
        cands = [0.0, 0.04, 0.08, 0.12]
    elif var == "humidity":
        cands = [0.0, 0.06, 0.12, 0.18, 0.24]
    elif var == "precip":
        cands = [0.0, 0.03, 0.06, 0.09, 0.12]
    else:
        cands = [0.0, 0.05, 0.10, 0.15]

    rec_gap = float(coverage.get("recency_gap_years", 0.0))
    cov_ratio = float(coverage.get("coverage_ratio", 1.0))
    add = 0.0
    if rec_gap > 6.0:
        add += 0.02
    if rec_gap > 12.0:
        add += 0.02
    if cov_ratio < 0.50:
        add += 0.02
    if add > 0:
        cands.append(min(0.30, max(cands) + add))
    cands = sorted(set(float(np.clip(c, 0.0, 0.35)) for c in cands))
    return cands


def apply_continuity_smoothing(
    blend: pd.DataFrame,
    variable: str,
    coverage: dict[str, Any],
    alpha_override: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": False,
        "reason": "not_applied",
        "alpha_override": (float(alpha_override) if alpha_override is not None else None),
        "selected_alpha": 0.0,
        "passes": 0,
        "before": {},
        "after": {},
        "table": [],
    }
    if blend is None or blend.empty:
        stats["reason"] = "empty_blend"
        return blend, stats
    if len(blend) < 12:
        stats["reason"] = "too_short"
        return blend, stats

    out = blend.copy()
    ds = pd.to_datetime(out["ds"], errors="coerce")
    y0 = pd.to_numeric(out["yhat"], errors="coerce").values.astype(float)
    lo0 = pd.to_numeric(out["yhat_lower"], errors="coerce").values.astype(float)
    hi0 = pd.to_numeric(out["yhat_upper"], errors="coerce").values.astype(float)
    if not np.isfinite(y0).all():
        stats["reason"] = "nonfinite_yhat"
        return blend, stats

    base = continuity_metrics(ds, y0)
    base_amp = float(max(base.get("amp_p90_p10", 0.0), 1e-6))
    base_mean = float(base.get("mean", 0.0))
    if not np.isfinite(base.get("jump_q95", np.nan)):
        stats["reason"] = "invalid_base_metrics"
        return blend, stats

    var = canonical_variable_name(variable)
    if var == "temp":
        w_ann, w_max = 0.45, 0.02
        mean_tol = max(0.25, 0.02 * abs(base_mean))
    elif var == "humidity":
        w_ann, w_max = 0.55, 0.03
        mean_tol = max(0.90, 0.015 * abs(base_mean))
    elif var == "precip":
        w_ann, w_max = 0.35, 0.01
        mean_tol = max(2.00, 0.03 * abs(base_mean))
    else:
        w_ann, w_max = 0.45, 0.02
        mean_tol = max(0.50, 0.03 * abs(base_mean))

    def obj_of(m: dict[str, float]) -> float:
        amp_pen = abs(np.log((float(m.get("amp_p90_p10", 0.0)) + 1e-6) / base_amp))
        return (
            float(m.get("jump_q95", 0.0))
            + w_ann * float(m.get("ann_jump_q95", 0.0))
            + w_max * float(m.get("jump_max", 0.0))
            + 0.05 * base_amp * float(amp_pen)
        )

    base_obj = float(obj_of(base))
    best_obj = base_obj
    best_alpha = 0.0
    best_y = y0.copy()
    best_m = dict(base)
    cand_rows: list[dict[str, Any]] = []

    jq95_lim = float(base.get("jump_q95", 0.0) * 1.01 + 1e-9)
    jmax_lim = float(base.get("jump_max", 0.0) * 1.05 + 1e-9)
    aq95_lim = float(base.get("ann_jump_q95", 0.0) * 1.10 + 1e-9)

    if alpha_override is None:
        alpha_list = continuity_alpha_candidates(variable=variable, coverage=coverage)
        alpha_source = "auto_candidates"
    else:
        ao = float(np.clip(float(alpha_override), 0.0, 0.35))
        alpha_list = sorted(set([0.0, ao]))
        alpha_source = "override_with_baseline"

    for a in alpha_list:
        passes = 3 if a >= 0.16 else 2
        ys = median3_smooth(y0, alpha=float(a), passes=passes)
        m = continuity_metrics(ds, ys)
        obj = float(obj_of(m))
        ok = (
            float(m.get("jump_q95", float("inf"))) <= jq95_lim
            and float(m.get("jump_max", float("inf"))) <= jmax_lim
            and float(m.get("ann_jump_q95", float("inf"))) <= aq95_lim
            and float(m.get("amp_p90_p10", 0.0)) >= 0.80 * base_amp
            and abs(float(m.get("mean", 0.0)) - base_mean) <= mean_tol
        )
        cand_rows.append(
            {
                "alpha": float(a),
                "passes": int(passes),
                "objective": float(obj),
                "eligible": bool(ok),
                "jump_q95": float(m.get("jump_q95", np.nan)),
                "jump_max": float(m.get("jump_max", np.nan)),
                "ann_jump_q95": float(m.get("ann_jump_q95", np.nan)),
                "ann_jump_max": float(m.get("ann_jump_max", np.nan)),
                "amp_p90_p10": float(m.get("amp_p90_p10", np.nan)),
                "mean": float(m.get("mean", np.nan)),
            }
        )
        if ok and obj < best_obj - 1e-9:
            best_obj = obj
            best_alpha = float(a)
            best_y = ys
            best_m = dict(m)

    if best_alpha <= 1e-9:
        rs = "no_better_candidate"
        if alpha_override is not None:
            rs = "override_rejected_or_not_better"
        stats.update(
            {
                "enabled": False,
                "reason": rs,
                "alpha_source": alpha_source,
                "selected_alpha": 0.0,
                "passes": 0,
                "before": {k: float(v) for k, v in base.items()},
                "after": {k: float(v) for k, v in base.items()},
                "table": cand_rows,
            }
        )
        return out, stats

    shift = best_y - y0
    out["yhat"] = best_y
    out["yhat_lower"] = lo0 + shift
    out["yhat_upper"] = hi0 + shift
    stats.update(
        {
            "enabled": True,
            "reason": "ok",
            "alpha_source": alpha_source,
            "selected_alpha": float(best_alpha),
            "passes": int(3 if best_alpha >= 0.16 else 2),
            "before": {k: float(v) for k, v in base.items()},
            "after": {k: float(v) for k, v in best_m.items()},
            "mean_abs_shift": float(np.mean(np.abs(shift))),
            "max_abs_shift": float(np.max(np.abs(shift))),
            "table": cand_rows,
        }
    )
    return out, stats


def build_climo_decay_member(
    variable: str,
    obs_monthly: pd.Series,
    forecast_start_year: int,
    target_year: int,
    interval_z: float,
    coverage: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": False,
        "reason": "not_applied",
        "rho": 0.0,
        "trend_per_year": 0.0,
        "sigma": 0.0,
        "recency_gap_years": float(coverage.get("recency_gap_years", 0.0)),
    }
    s = pd.to_numeric(obs_monthly, errors="coerce").dropna().sort_index()
    if s.empty:
        stats["reason"] = "empty_observations"
        return pd.DataFrame(), stats

    month_clim = smooth_month_climatology(s, variable=variable)
    if len(month_clim) < 6:
        stats["reason"] = "insufficient_climatology"
        return pd.DataFrame(), stats

    recency_gap_years = float(coverage.get("recency_gap_years", 0.0))
    t_hist = (s.index.year.values.astype(float) + (s.index.month.values.astype(float) - 0.5) / 12.0) - float(
        max(int(forecast_start_year) - 1, 1)
    )

    trend = 0.0
    if len(s) >= 24:
        ann = s.groupby(s.index.year).mean().dropna()
        if len(ann) >= 2:
            years_span = float(max(int(ann.index.max()) - int(ann.index.min()), 1))
            trend = float((ann.iloc[-1] - ann.iloc[0]) / years_span)
    trend *= float(np.clip(1.0 - recency_gap_years / 25.0, 0.10, 1.0))

    clim_hist = np.array([float(month_clim.get(int(m), np.nan)) for m in s.index.month.values], dtype=float)
    base_hist = clim_hist + trend * t_hist
    anom = s.values.astype(float) - base_hist

    rho = 0.0
    if len(anom) >= 3:
        x = anom[:-1]
        y = anom[1:]
        ok = np.isfinite(x) & np.isfinite(y)
        if int(ok.sum()) >= 2:
            den = float(np.dot(x[ok], x[ok])) + 1e-8
            rho = float(np.dot(x[ok], y[ok]) / den)
    rho = float(np.clip(rho, -0.25, 0.88))
    rho *= float(np.clip(1.0 - recency_gap_years / 18.0, 0.18, 1.0))

    sigma = float(np.nanstd(anom))
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = float(np.nanstd(s.values.astype(float)) * 0.18)
    sigma = float(max(sigma, 0.15 if variable == "temp" else 0.8))

    month_resid = {}
    for m in range(1, 13):
        rr = anom[s.index.month.values == m]
        rr = rr[np.isfinite(rr)]
        if len(rr) >= 3:
            month_resid[m] = float(np.nanstd(rr))
        else:
            month_resid[m] = sigma

    yhat_hist = base_hist.copy()
    ylo_hist = yhat_hist - float(interval_z) * np.array([month_resid.get(int(m), sigma) for m in s.index.month.values], dtype=float)
    yhi_hist = yhat_hist + float(interval_z) * np.array([month_resid.get(int(m), sigma) for m in s.index.month.values], dtype=float)

    start_anom = float(anom[-1]) if len(anom) else 0.0
    start_anom *= float(np.exp(-max(recency_gap_years, 0.0) / 3.0))
    if recency_gap_years >= 12.0:
        start_anom *= 0.35

    f_idx = pd.date_range(f"{int(forecast_start_year)}-01-01", f"{int(target_year)}-12-01", freq="MS")
    if len(f_idx) == 0:
        stats["reason"] = "empty_forecast_horizon"
        return pd.DataFrame(), stats

    yf = np.zeros(len(f_idx), dtype=float)
    ylf = np.zeros(len(f_idx), dtype=float)
    yhf = np.zeros(len(f_idx), dtype=float)
    a_prev = float(start_anom)
    for i, ts in enumerate(f_idx):
        m = int(ts.month)
        t = (float(ts.year) + (float(ts.month) - 0.5) / 12.0) - float(max(int(forecast_start_year) - 1, 1))
        clim = float(month_clim.get(m, np.nanmean(list(month_clim.values()))))
        a_prev = float(rho * a_prev)
        yv = clim + trend * t + a_prev
        sig_m = float(month_resid.get(m, sigma))
        sig_h = sig_m * float(np.clip(1.0 + 0.01 * i, 1.0, 1.8))
        yf[i] = yv
        ylf[i] = yv - float(interval_z) * sig_h
        yhf[i] = yv + float(interval_z) * sig_h

    yf = apply_bounds(yf, variable)
    ylf = apply_bounds(ylf, variable)
    yhf = apply_bounds(yhf, variable)
    ylf = np.minimum(ylf, yf)
    yhf = np.maximum(yhf, yf)
    yhat_hist = apply_bounds(yhat_hist, variable)
    ylo_hist = apply_bounds(ylo_hist, variable)
    yhi_hist = apply_bounds(yhi_hist, variable)
    ylo_hist = np.minimum(ylo_hist, yhat_hist)
    yhi_hist = np.maximum(yhi_hist, yhat_hist)

    hist = pd.DataFrame(
        {
            "ds": pd.to_datetime(s.index),
            "actual": s.values.astype(float),
            "yhat": yhat_hist.astype(float),
            "yhat_lower": ylo_hist.astype(float),
            "yhat_upper": yhi_hist.astype(float),
            "is_forecast": False,
            "mean_reversion": 0.0,
            "climate_delta": 0.0,
        }
    )
    fc = pd.DataFrame(
        {
            "ds": pd.to_datetime(f_idx),
            "actual": np.nan,
            "yhat": yf.astype(float),
            "yhat_lower": ylf.astype(float),
            "yhat_upper": yhf.astype(float),
            "is_forecast": True,
            "mean_reversion": 0.0,
            "climate_delta": 0.0,
        }
    )
    out = pd.concat([hist, fc], ignore_index=True).sort_values("ds").reset_index(drop=True)
    out["variable"] = variable
    out["unit"] = infer_unit(variable)
    out["frequency"] = "MS"
    out["model_strategy"] = "climo_decay_baseline"

    stats.update(
        {
            "enabled": True,
            "reason": "ok",
            "rho": float(rho),
            "trend_per_year": float(trend),
            "sigma": float(sigma),
            "start_anomaly": float(start_anom),
        }
    )
    return out, stats


def build_virtual_member_variants(
    variable: str,
    obs_monthly: pd.Series,
    base_members: list[tuple[ModelScore, pd.DataFrame]],
) -> tuple[list[tuple[ModelScore, pd.DataFrame]], dict[str, Any]]:
    month_clim = smooth_month_climatology(pd.to_numeric(obs_monthly, errors="coerce").dropna().sort_index(), variable=variable)
    out: list[tuple[ModelScore, pd.DataFrame]] = []
    stats: dict[str, Any] = {
        "enabled": False,
        "base_member_count": int(len(base_members)),
        "virtual_member_count": 0,
        "virtual_model_keys": [],
    }
    if not base_members or len(month_clim) < 6:
        stats["reason"] = "insufficient_base_or_climatology"
        return out, stats

    def clone_score(ms: ModelScore, key: str, rmse_mult: float, bias_mult: float, std_mult: float, variant_name: str) -> ModelScore:
        rm = safe_float(ms.rmse, default=np.nan)
        bi = safe_float(ms.bias, default=0.0)
        st = safe_float(ms.rmse_std, default=0.0)
        if np.isfinite(rm):
            rm = float(max(1e-6, rm * rmse_mult))
        else:
            rm = np.nan
        return ModelScore(
            model_key=key,
            variable=ms.variable,
            frequency=ms.frequency,
            forecast_csv=f"generated://{key}",
            rmse=rm,
            bias=float(bi * bias_mult),
            rmse_std=float(max(0.0, st * std_mult)),
            has_cv=False,
            source=f"{ms.source}::virtual",
            raw={**(ms.raw or {}), "virtual_variant": variant_name, "parent_model_key": ms.model_key},
        )

    for ms, dfm in base_members:
        d0 = dfm.copy().sort_values("ds").reset_index(drop=True)
        fc_mask = d0["is_forecast"] == True
        if not fc_mask.any():
            continue

        # Variant-1: EWMA smooth forecast path.
        d1 = d0.copy()
        alpha = 0.50 if variable == "temp" else (0.42 if variable == "humidity" else 0.35)
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            if col not in d1.columns:
                continue
            s = pd.to_numeric(d1.loc[fc_mask, col], errors="coerce")
            if s.notna().sum() == 0:
                continue
            d1.loc[fc_mask, col] = s.ewm(alpha=alpha, adjust=False).mean().values
        d1["yhat"] = apply_bounds(pd.to_numeric(d1["yhat"], errors="coerce").values, variable)
        d1["yhat_lower"] = apply_bounds(pd.to_numeric(d1["yhat_lower"], errors="coerce").values, variable)
        d1["yhat_upper"] = apply_bounds(pd.to_numeric(d1["yhat_upper"], errors="coerce").values, variable)
        d1["yhat_lower"] = np.minimum(d1["yhat_lower"], d1["yhat"])
        d1["yhat_upper"] = np.maximum(d1["yhat_upper"], d1["yhat"])
        key1 = f"{ms.model_key}__v_smooth"
        ms1 = clone_score(ms, key=key1, rmse_mult=1.05, bias_mult=1.00, std_mult=0.92, variant_name="ewma_smooth")
        out.append((ms1, d1))
        stats["virtual_model_keys"].append(key1)

        # Variant-2: Blend with monthly climatology anchor.
        d2 = d0.copy()
        w_clim = 0.42 if variable == "temp" else (0.30 if variable == "humidity" else 0.48)
        f2 = d2.loc[fc_mask].copy()
        if not f2.empty:
            anchors = np.array([float(month_clim.get(int(pd.Timestamp(ts).month), np.nan)) for ts in f2["ds"].tolist()], dtype=float)
            y_old = pd.to_numeric(f2["yhat"], errors="coerce").values.astype(float)
            y_new = (1.0 - w_clim) * y_old + w_clim * anchors
            shift = y_new - y_old
            f2["yhat"] = y_new
            if "yhat_lower" in f2.columns:
                f2["yhat_lower"] = pd.to_numeric(f2["yhat_lower"], errors="coerce").values.astype(float) + shift
            if "yhat_upper" in f2.columns:
                f2["yhat_upper"] = pd.to_numeric(f2["yhat_upper"], errors="coerce").values.astype(float) + shift
            d2.loc[f2.index, ["yhat", "yhat_lower", "yhat_upper"]] = f2[["yhat", "yhat_lower", "yhat_upper"]].values
        d2["yhat"] = apply_bounds(pd.to_numeric(d2["yhat"], errors="coerce").values, variable)
        d2["yhat_lower"] = apply_bounds(pd.to_numeric(d2["yhat_lower"], errors="coerce").values, variable)
        d2["yhat_upper"] = apply_bounds(pd.to_numeric(d2["yhat_upper"], errors="coerce").values, variable)
        d2["yhat_lower"] = np.minimum(d2["yhat_lower"], d2["yhat"])
        d2["yhat_upper"] = np.maximum(d2["yhat_upper"], d2["yhat"])
        key2 = f"{ms.model_key}__v_climo"
        ms2 = clone_score(ms, key=key2, rmse_mult=1.07, bias_mult=0.92, std_mult=0.95, variant_name="climatology_blend")
        out.append((ms2, d2))
        stats["virtual_model_keys"].append(key2)

        # Variant-3: Horizon-damped conservative path (pull toward climatology with horizon).
        d3 = d0.copy()
        f3 = d3.loc[fc_mask].copy()
        if not f3.empty:
            anchors = np.array([float(month_clim.get(int(pd.Timestamp(ts).month), np.nan)) for ts in f3["ds"].tolist()], dtype=float)
            y_old = pd.to_numeric(f3["yhat"], errors="coerce").values.astype(float)
            n = len(y_old)
            if n > 1:
                t = np.linspace(0.0, 1.0, n, dtype=float)
            else:
                t = np.array([1.0], dtype=float)
            if variable == "temp":
                w0, w1 = 0.25, 0.72
            elif variable == "humidity":
                w0, w1 = 0.22, 0.62
            elif variable == "precip":
                w0, w1 = 0.30, 0.78
            else:
                w0, w1 = 0.20, 0.60
            w = np.clip(w0 + (w1 - w0) * t, w0, w1)
            y_mid = (1.0 - w) * y_old + w * anchors
            y_new = pd.Series(y_mid).ewm(alpha=0.38, adjust=False).mean().values
            shift = y_new - y_old
            f3["yhat"] = y_new
            if "yhat_lower" in f3.columns:
                f3["yhat_lower"] = pd.to_numeric(f3["yhat_lower"], errors="coerce").values.astype(float) + shift
            if "yhat_upper" in f3.columns:
                f3["yhat_upper"] = pd.to_numeric(f3["yhat_upper"], errors="coerce").values.astype(float) + shift
            d3.loc[f3.index, ["yhat", "yhat_lower", "yhat_upper"]] = f3[["yhat", "yhat_lower", "yhat_upper"]].values
        d3["yhat"] = apply_bounds(pd.to_numeric(d3["yhat"], errors="coerce").values, variable)
        d3["yhat_lower"] = apply_bounds(pd.to_numeric(d3["yhat_lower"], errors="coerce").values, variable)
        d3["yhat_upper"] = apply_bounds(pd.to_numeric(d3["yhat_upper"], errors="coerce").values, variable)
        d3["yhat_lower"] = np.minimum(d3["yhat_lower"], d3["yhat"])
        d3["yhat_upper"] = np.maximum(d3["yhat_upper"], d3["yhat"])
        key3 = f"{ms.model_key}__v_damped"
        ms3 = clone_score(ms, key=key3, rmse_mult=1.09, bias_mult=0.90, std_mult=0.88, variant_name="horizon_damped_conservative")
        out.append((ms3, d3))
        stats["virtual_model_keys"].append(key3)

    stats["enabled"] = True
    stats["reason"] = "ok"
    stats["virtual_member_count"] = int(len(out))
    return out, stats


def coupled_temp_from_humidity_forecast(
    obs: pd.DataFrame,
    temp_df: pd.DataFrame,
    humidity_df: pd.DataFrame,
    ok_value: str,
    forecast_start_year: int,
    target_year: int,
    interval_z: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {
        "enabled": False,
        "reason": "not_applied",
        "overlap_months": 0,
        "blend_weight_coupled": 0.0,
        "train_rmse_c": None,
    }
    if temp_df is None or humidity_df is None:
        stats["reason"] = "missing_temp_or_humidity_forecast"
        return temp_df, stats

    temp_obs = monthly_from_obs_no_fill(obs, variable="temp", ok_value=ok_value)
    hum_obs = monthly_from_obs_no_fill(obs, variable="humidity", ok_value=ok_value)
    if temp_obs.empty or hum_obs.empty:
        stats["reason"] = "missing_temp_or_humidity_observations"
        return temp_df, stats

    ref_end = pd.Timestamp(year=max(int(forecast_start_year) - 1, 1), month=12, day=1)
    cov_temp = monthly_history_coverage(temp_obs, reference_end=ref_end)
    cov_ratio = float(cov_temp.get("coverage_ratio", 0.0))
    recency_gap_years = float(cov_temp.get("recency_gap_years", 0.0))
    # Apply only when temperature history is sparse.
    if int(cov_temp["months_with_data"]) >= 72 and float(cov_temp["coverage_ratio"]) >= 0.35:
        stats["reason"] = "temp_history_not_sparse"
        return temp_df, stats

    idx = temp_obs.index.union(hum_obs.index).sort_values()
    t = temp_obs.reindex(idx)
    h = hum_obs.reindex(idx)
    overlap = t.notna() & h.notna()
    n_overlap = int(overlap.sum())
    stats["overlap_months"] = n_overlap
    if n_overlap < 10:
        stats["reason"] = "insufficient_overlap"
        return temp_df, stats

    train_idx = idx[overlap.values]
    y = t.loc[train_idx].values.astype(float)
    hv = h.loc[train_idx].values.astype(float)

    def design(ix: pd.DatetimeIndex, mode: str, hum_vals: np.ndarray | None = None) -> np.ndarray:
        m = ix.month.values.astype(float)
        sin_m = np.sin(2.0 * np.pi * (m - 1.0) / 12.0)
        cos_m = np.cos(2.0 * np.pi * (m - 1.0) / 12.0)
        if mode == "month_only":
            return np.column_stack([np.ones(len(ix)), sin_m, cos_m])
        if mode == "hum_only":
            if hum_vals is None:
                raise ValueError("hum_vals required for hum_only mode")
            return np.column_stack([np.ones(len(ix)), hum_vals])
        if mode == "hum_month":
            if hum_vals is None:
                raise ValueError("hum_vals required for hum_month mode")
            return np.column_stack([np.ones(len(ix)), hum_vals, sin_m, cos_m])
        raise ValueError(f"Unknown design mode: {mode}")

    def loocv_rmse_ridge(X: np.ndarray, yy: np.ndarray, lam: float = 1.0) -> float:
        n = len(yy)
        if n < 4:
            return float("inf")
        errs = []
        for i in range(n):
            tr = np.ones(n, dtype=bool)
            tr[i] = False
            Xtr = X[tr]
            ytr = yy[tr]
            reg = np.eye(Xtr.shape[1], dtype=float) * lam
            reg[0, 0] = 0.0
            try:
                b = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
            except np.linalg.LinAlgError:
                return float("inf")
            yp = float(X[i] @ b)
            errs.append((yy[i] - yp) ** 2)
        return float(np.sqrt(np.mean(errs)))

    X_month = design(train_idx, mode="month_only")
    X_hum = design(train_idx, mode="hum_only", hum_vals=hv)
    X_hm = design(train_idx, mode="hum_month", hum_vals=hv)

    rmse_candidates = {
        "month_only": loocv_rmse_ridge(X_month, y, lam=1.0),
        "hum_only": loocv_rmse_ridge(X_hum, y, lam=1.0),
        "hum_month": loocv_rmse_ridge(X_hm, y, lam=1.0),
    }
    best_mode = min(rmse_candidates, key=rmse_candidates.get)
    best_rmse = float(rmse_candidates[best_mode])
    if not np.isfinite(best_rmse):
        stats["reason"] = "cv_failed"
        return temp_df, stats

    mode_override_reason = None
    # Very old and sparse temp history: prefer humidity-coupled mode when not much worse.
    if recency_gap_years >= 15.0 and cov_ratio < 0.12:
        rm_hm = rmse_candidates.get("hum_month", float("inf"))
        rm_h = rmse_candidates.get("hum_only", float("inf"))
        if np.isfinite(rm_hm) and rm_hm <= best_rmse * 1.45:
            best_mode = "hum_month"
            best_rmse = float(rm_hm)
            mode_override_reason = "forced_humidity_coupling_sparse_stale_temp"
        elif np.isfinite(rm_h) and rm_h <= best_rmse * 1.35:
            best_mode = "hum_only"
            best_rmse = float(rm_h)
            mode_override_reason = "forced_humidity_only_sparse_stale_temp"

    X_best = {
        "month_only": X_month,
        "hum_only": X_hum,
        "hum_month": X_hm,
    }[best_mode]
    reg = np.eye(X_best.shape[1], dtype=float) * 1.0
    reg[0, 0] = 0.0
    beta = np.linalg.solve(X_best.T @ X_best + reg, X_best.T @ y)
    y_fit = X_best @ beta
    rmse = float(np.sqrt(np.mean((y - y_fit) ** 2)))
    stats["train_rmse_c"] = rmse
    stats["cv_mode_selected"] = best_mode
    stats["cv_rmse_candidates"] = {k: (float(v) if np.isfinite(v) else None) for k, v in rmse_candidates.items()}
    if mode_override_reason is not None:
        stats["cv_mode_override_reason"] = mode_override_reason

    tf = temp_df.copy()
    tf["ds"] = pd.to_datetime(tf["ds"], errors="coerce")
    tf_fc = tf[(tf["is_forecast"] == True)].copy()
    tf_fc = tf_fc[(tf_fc["ds"].dt.year >= int(forecast_start_year)) & (tf_fc["ds"].dt.year <= int(target_year))]
    if tf_fc.empty:
        stats["reason"] = "empty_forecast_window"
        return temp_df, stats

    tf_fc = tf_fc.set_index("ds").sort_index()
    hf_fc = None
    if best_mode in {"hum_only", "hum_month"}:
        hf = humidity_df.copy()
        hf["ds"] = pd.to_datetime(hf["ds"], errors="coerce")
        hf_fc = hf[(hf["is_forecast"] == True)].copy()
        hf_fc = hf_fc[(hf_fc["ds"].dt.year >= int(forecast_start_year)) & (hf_fc["ds"].dt.year <= int(target_year))]
        if hf_fc.empty:
            stats["reason"] = "empty_humidity_forecast_for_hum_mode"
            return temp_df, stats
        hf_fc = hf_fc.set_index("ds").sort_index()
        common = tf_fc.index.intersection(hf_fc.index)
        if len(common) < 6:
            stats["reason"] = "insufficient_common_forecast_months"
            return temp_df, stats
        h_pred = hf_fc.loc[common, "yhat"].values.astype(float)
        Xp = design(pd.DatetimeIndex(common), mode=best_mode, hum_vals=h_pred)
    else:
        common = tf_fc.index
        Xp = design(pd.DatetimeIndex(common), mode=best_mode)
    y_cpl = Xp @ beta

    ex_y = tf_fc.loc[common, "yhat"].values.astype(float)
    correction_raw = y_cpl - ex_y
    correction_smoothed = correction_raw.copy()
    correction_alpha = 0.0
    correction_passes = 0
    if len(correction_raw) >= 24 and np.isfinite(correction_raw).all():
        # Smooth humidity-coupling correction itself to avoid introducing
        # month-to-month kinks when humidity trajectory has local spikes.
        smooth_strength = float(np.clip((1.0 - cov_ratio) * 0.85 + recency_gap_years / 25.0, 0.0, 1.0))
        correction_alpha = float(np.clip(0.10 + 0.22 * smooth_strength, 0.10, 0.32))
        correction_passes = int(3 if correction_alpha >= 0.20 else 2)
        correction_smoothed = median3_smooth(correction_raw, alpha=correction_alpha, passes=correction_passes)
        raw_mean = float(np.nanmean(correction_raw))
        sm_mean = float(np.nanmean(correction_smoothed))
        if np.isfinite(raw_mean) and np.isfinite(sm_mean):
            correction_smoothed = correction_smoothed + (raw_mean - sm_mean)
    y_cpl_eff = ex_y + correction_smoothed

    hum_half = np.zeros(len(common), dtype=float)
    if hf_fc is not None and {"yhat_lower", "yhat_upper"}.issubset(hf_fc.columns):
        hh = (hf_fc.loc[common, "yhat_upper"].values.astype(float) - hf_fc.loc[common, "yhat_lower"].values.astype(float)) / 2.0
        hum_half = np.clip(hh, 0.0, None)
    hum_coef_abs = float(abs(beta[1])) if best_mode in {"hum_only", "hum_month"} else 0.0
    spread = float(interval_z) * rmse + hum_coef_abs * hum_half
    c_lo = y_cpl_eff - spread
    c_hi = y_cpl_eff + spread

    coverage_ratio = float(cov_temp["coverage_ratio"])
    coupled_w = float(np.clip(0.40 + (1.0 - coverage_ratio) * 0.45, 0.40, 0.88))
    if n_overlap < 16:
        coupled_w *= 0.88
    if best_mode == "month_only":
        coupled_w = min(0.92, coupled_w + 0.07)
    elif best_mode == "hum_only":
        coupled_w = max(0.30, coupled_w - 0.08)
    coupled_w = float(np.clip(coupled_w, 0.30, 0.88))
    stats["blend_weight_coupled"] = coupled_w

    ex_lo = tf_fc.loc[common, "yhat_lower"].values.astype(float)
    ex_hi = tf_fc.loc[common, "yhat_upper"].values.astype(float)

    ny = (1.0 - coupled_w) * ex_y + coupled_w * y_cpl_eff
    nlo = (1.0 - coupled_w) * ex_lo + coupled_w * c_lo
    nhi = (1.0 - coupled_w) * ex_hi + coupled_w * c_hi

    ny = apply_bounds(ny, "temp")
    nlo = apply_bounds(nlo, "temp")
    nhi = apply_bounds(nhi, "temp")
    nlo = np.minimum(nlo, ny)
    nhi = np.maximum(nhi, ny)

    tf_fc.loc[common, "yhat"] = ny
    tf_fc.loc[common, "yhat_lower"] = nlo
    tf_fc.loc[common, "yhat_upper"] = nhi
    tf_fc["model_strategy"] = "best_meta_ensemble_coupled_humidity"

    tf_hist = tf[tf["is_forecast"] == False].copy()
    out = pd.concat([tf_hist, tf_fc.reset_index()], ignore_index=True).sort_values("ds").reset_index(drop=True)
    stats.update(
        {
            "enabled": True,
            "reason": "ok",
            "coef_intercept": float(beta[0]),
            "coef_humidity": float(beta[1]) if best_mode in {"hum_only", "hum_month"} else 0.0,
            "coef_sin_month": float(beta[-2]) if best_mode in {"month_only", "hum_month"} else 0.0,
            "coef_cos_month": float(beta[-1]) if best_mode in {"month_only", "hum_month"} else 0.0,
            "coupling_correction_smoothing_alpha": float(correction_alpha),
            "coupling_correction_smoothing_passes": int(correction_passes),
            "coupling_correction_mean_abs_raw": float(np.mean(np.abs(correction_raw))) if len(correction_raw) else 0.0,
            "coupling_correction_mean_abs_smoothed": float(np.mean(np.abs(correction_smoothed))) if len(correction_smoothed) else 0.0,
        }
    )
    return out, stats


def run_command(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        tail_out = "\n".join((p.stdout or "").splitlines()[-30:])
        tail_err = "\n".join((p.stderr or "").splitlines()[-30:])
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"stdout_tail:\n{tail_out}\n\nstderr_tail:\n{tail_err}"
        )


def maybe_run_base_models(args: argparse.Namespace, base_dir: Path) -> None:
    if not to_bool(args.run_base_models):
        return

    base_dir.mkdir(parents=True, exist_ok=True)
    quant_dir = base_dir / "quant"
    strong_dir = base_dir / "strong"
    ultra_dir = base_dir / "prophet_ultra"
    wf_dir = base_dir / "walkforward"

    quant_idx = quant_dir / f"quant_index_to_{args.target_year}.csv"
    strong_idx = strong_dir / f"strong_ensemble_index_to_{args.target_year}.csv"
    ultra_idx = ultra_dir / f"prophet_ultra_index_to_{args.target_year}.csv"
    wf_idx = wf_dir / f"walkforward_index_{args.walkforward_start_year}_{args.target_year}.csv"

    if not (to_bool(args.reuse_existing) and quant_idx.exists()):
        run_command(
            [
                "python3",
                "scripts/quant_regime_projection.py",
                "--observations",
                str(args.observations),
                "--output-dir",
                str(quant_dir),
                "--variables",
                str(args.variables),
                "--target-year",
                str(args.target_year),
                "--climate-scenario",
                str(args.climate_scenario),
                "--climate-baseline-year",
                str(args.climate_baseline_year),
                "--climate-temp-rate",
                str(args.climate_temp_rate),
                "--humidity-per-temp-c",
                str(args.humidity_per_temp_c),
                "--climate-adjustment-method",
                str(args.climate_adjustment_method),
            ]
            + (["--disable-climate-adjustment"] if args.disable_climate_adjustment else []),
            cwd=Path.cwd(),
        )

    if not (to_bool(args.reuse_existing) and strong_idx.exists()):
        run_command(
            [
                "python3",
                "scripts/train_strong_consistent_model.py",
                "--observations",
                str(args.observations),
                "--output-dir",
                str(strong_dir),
                "--variables",
                str(args.variables),
                "--target-year",
                str(args.target_year),
            ],
            cwd=Path.cwd(),
        )

    if not (to_bool(args.reuse_existing) and ultra_idx.exists()):
        run_command(
            [
                "python3",
                "scripts/prophet_ultra_500.py",
                "--observations",
                str(args.observations),
                "--output-dir",
                str(ultra_dir),
                "--variables",
                str(args.variables),
                "--target-year",
                str(args.target_year),
                "--input-kind",
                "auto",
            ],
            cwd=Path.cwd(),
        )

    if not (to_bool(args.reuse_existing) and wf_idx.exists()):
        run_command(
            [
                "python3",
                "scripts/walkforward_retrain_multifreq.py",
                "--observations",
                str(args.observations),
                "--output-dir",
                str(wf_dir),
                "--variables",
                str(args.variables),
                "--frequencies",
                "MS",
                "--start-year",
                str(args.walkforward_start_year),
                "--target-year",
                str(args.target_year),
                "--input-kind",
                "auto",
                "--climate-scenario",
                str(args.climate_scenario),
                "--climate-baseline-year",
                str(args.climate_baseline_year),
                "--climate-temp-rate",
                str(args.climate_temp_rate),
                "--humidity-per-temp-c",
                str(args.humidity_per_temp_c),
                "--climate-adjustment-method",
                str(args.climate_adjustment_method),
            ]
            + (["--disable-climate-adjustment"] if args.disable_climate_adjustment else []),
            cwd=Path.cwd(),
        )


def resolve_path(path_like: str, root: Path) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return (root / p).resolve()


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def member_is_plausible(
    variable: str,
    obs_monthly: pd.Series,
    dfm: pd.DataFrame,
    forecast_start_year: int,
    target_year: int,
) -> bool:
    fc = dfm[dfm["is_forecast"] == True].copy()
    if fc.empty:
        return False
    fc = fc[(fc["ds"].dt.year >= int(forecast_start_year)) & (fc["ds"].dt.year <= int(target_year))].copy()
    if fc.empty:
        return False
    y = pd.to_numeric(fc["yhat"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
    if len(y) == 0:
        return False

    var = canonical_variable_name(variable)
    if var == "temp":
        # Hard plausibility range for monthly mean temperature (global-safe).
        if np.max(np.abs(y)) > 120.0:
            return False
        # Scale-sensitive divergence check vs observed history.
        obs_std = float(np.nanstd(obs_monthly.values)) if len(obs_monthly) else 1.0
        obs_abs_med = float(np.nanmedian(np.abs(obs_monthly.values))) if len(obs_monthly) else 10.0
        obs_std = max(obs_std, 0.5)
        obs_abs_med = max(obs_abs_med, 1.0)
        y_std = float(np.nanstd(y))
        y_abs_q99 = float(np.nanquantile(np.abs(y), 0.99))
        if y_std > max(18.0, 8.0 * obs_std):
            return False
        if y_abs_q99 > max(45.0, 6.0 * obs_abs_med):
            return False
    elif var == "humidity":
        # Relative humidity should remain in [0, 100] for monthly means.
        if np.nanmin(y) < -5.0 or np.nanmax(y) > 105.0:
            return False
    return True


def load_model_scores(base_dir: Path, target_year: int, start_year: int) -> list[ModelScore]:
    out: list[ModelScore] = []

    # quant
    q_idx = base_dir / "quant" / f"quant_index_to_{target_year}.csv"
    if q_idx.exists():
        q = pd.read_csv(q_idx)
        for _, r in q.iterrows():
            out.append(
                ModelScore(
                    model_key="quant",
                    variable=canonical_variable_name(r.get("variable")),
                    frequency=str(r.get("frequency", "")),
                    forecast_csv=str(r.get("forecast_csv")),
                    rmse=safe_float(r.get("cv_rmse"), default=np.nan),
                    bias=safe_float(r.get("cv_bias"), default=0.0),
                    rmse_std=safe_float(r.get("cv_rmse_std"), default=0.0),
                    has_cv=np.isfinite(safe_float(r.get("cv_rmse"), default=np.nan)),
                    source=str(q_idx),
                    raw=r.to_dict(),
                )
            )

    # strong
    s_idx = base_dir / "strong" / f"strong_ensemble_index_to_{target_year}.csv"
    if s_idx.exists():
        s = pd.read_csv(s_idx)
        for _, r in s.iterrows():
            out.append(
                ModelScore(
                    model_key="strong",
                    variable=canonical_variable_name(r.get("variable")),
                    frequency=str(r.get("frequency", "")),
                    forecast_csv=str(r.get("forecast_csv")),
                    rmse=safe_float(r.get("best_cv_rmse"), default=np.nan),
                    bias=0.0,
                    rmse_std=0.0,
                    has_cv=np.isfinite(safe_float(r.get("best_cv_rmse"), default=np.nan)),
                    source=str(s_idx),
                    raw=r.to_dict(),
                )
            )

    # prophet ultra
    p_idx = base_dir / "prophet_ultra" / f"prophet_ultra_index_to_{target_year}.csv"
    if p_idx.exists():
        p = pd.read_csv(p_idx)
        for _, r in p.iterrows():
            out.append(
                ModelScore(
                    model_key="prophet_ultra",
                    variable=canonical_variable_name(r.get("variable")),
                    frequency=str(r.get("frequency", "")),
                    forecast_csv=str(r.get("forecast_csv")),
                    rmse=safe_float(r.get("best_cv_rmse"), default=np.nan),
                    bias=safe_float(r.get("bias_correction"), default=0.0),
                    rmse_std=0.0,
                    has_cv=np.isfinite(safe_float(r.get("best_cv_rmse"), default=np.nan)),
                    source=str(p_idx),
                    raw=r.to_dict(),
                )
            )

    # walkforward (no explicit CV metric)
    w_idx = base_dir / "walkforward" / f"walkforward_index_{start_year}_{target_year}.csv"
    if w_idx.exists():
        w = pd.read_csv(w_idx)
        for _, r in w.iterrows():
            forecast_path = resolve_path(str(r.get("forecast_csv")), Path.cwd())
            rmse = np.nan
            bias = 0.0
            std = 0.0
            if forecast_path.exists():
                wf = read_table(forecast_path)
                if "is_forecast" in wf.columns:
                    h = wf[wf["is_forecast"] == False].copy()
                else:
                    h = wf.copy()
                if {"actual", "yhat"}.issubset(set(h.columns)):
                    hh = h.dropna(subset=["actual", "yhat"])
                    if not hh.empty:
                        e = hh["actual"].values.astype(float) - hh["yhat"].values.astype(float)
                        rmse = float(np.sqrt(np.mean(e**2)))
                        bias = float(np.mean(e))
                        std = float(np.std(e))

            out.append(
                ModelScore(
                    model_key="walkforward",
                    variable=canonical_variable_name(r.get("variable")),
                    frequency=str(r.get("frequency", "")),
                    forecast_csv=str(r.get("forecast_csv")),
                    rmse=safe_float(rmse, default=np.nan),
                    bias=safe_float(bias, default=0.0),
                    rmse_std=safe_float(std, default=0.0),
                    has_cv=False,
                    source=str(w_idx),
                    raw=r.to_dict(),
                )
            )

    return out


def standardize_forecast_frame(
    path: Path,
    variable: str,
) -> pd.DataFrame:
    df = read_table(path)
    if "ds" not in df.columns:
        raise RuntimeError(f"'ds' missing in forecast: {path}")
    d = df.copy()
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d = d.dropna(subset=["ds"]).sort_values("ds")

    if "variable" in d.columns:
        d["variable"] = d["variable"].astype(str).map(canonical_variable_name)
        d = d[d["variable"] == variable].copy()

    if "yhat" not in d.columns:
        if "value" in d.columns:
            d["yhat"] = pd.to_numeric(d["value"], errors="coerce")
        else:
            raise RuntimeError(f"'yhat' missing in forecast: {path}")

    for c in ["yhat_lower", "yhat_upper"]:
        if c not in d.columns:
            d[c] = np.nan

    if "is_forecast" not in d.columns:
        # If unknown, treat all rows as forecast.
        d["is_forecast"] = True

    if "actual" not in d.columns:
        d["actual"] = np.nan

    d["yhat"] = pd.to_numeric(d["yhat"], errors="coerce")
    d["yhat_lower"] = pd.to_numeric(d["yhat_lower"], errors="coerce")
    d["yhat_upper"] = pd.to_numeric(d["yhat_upper"], errors="coerce")
    d["actual"] = pd.to_numeric(d["actual"], errors="coerce")
    d = d.dropna(subset=["yhat"])
    return d


def apply_climate_to_member(
    df: pd.DataFrame,
    variable: str,
    climate_cfg_global: ClimateAdjustmentConfig,
) -> pd.DataFrame:
    d = df.copy()
    if not climate_cfg_global.enabled:
        if "climate_delta" not in d.columns:
            d["climate_delta"] = 0.0
        return d

    has_delta = "climate_delta" in d.columns and d["climate_delta"].notna().any()
    if has_delta:
        d["climate_delta"] = pd.to_numeric(d["climate_delta"], errors="coerce").fillna(0.0)
        return d

    hist = d[d["is_forecast"] == False]
    if hist.empty:
        base_ts = pd.Timestamp(d["ds"].min())
    else:
        base_ts = pd.Timestamp(hist["ds"].max())
    cfg = with_series_baseline(climate_cfg_global, base_ts)

    d["climate_delta"] = 0.0
    fc_mask = d["is_forecast"] == True
    if fc_mask.any():
        delta = climate_delta_series(d.loc[fc_mask, "ds"], variable=variable, cfg=cfg)
        d.loc[fc_mask, "climate_delta"] = delta
        d.loc[fc_mask, "yhat"] = d.loc[fc_mask, "yhat"].values + delta
        if d["yhat_lower"].notna().any():
            d.loc[fc_mask, "yhat_lower"] = d.loc[fc_mask, "yhat_lower"].values + delta
        if d["yhat_upper"].notna().any():
            d.loc[fc_mask, "yhat_upper"] = d.loc[fc_mask, "yhat_upper"].values + delta

    return d


def model_score_value(
    ms: ModelScore,
    obs_scale: float,
    variable: str,
    history_months: int,
    history_years: int,
    coverage_ratio: float,
    recency_gap_years: float,
) -> float:
    rmse = safe_float(ms.rmse, default=np.nan)
    if not np.isfinite(rmse):
        rmse = max(0.5, 0.12 * max(1.0, obs_scale))

    bias = safe_float(ms.bias, default=0.0)
    rmse_std = safe_float(ms.rmse_std, default=0.0)
    bias_norm = abs(bias) / max(1.0, 0.08 * obs_scale)
    stab_norm = rmse_std / max(rmse, 1e-6)
    score = rmse * (1.0 + 0.35 * bias_norm + 0.20 * stab_norm)

    # Penalize non-CV model metrics.
    if not ms.has_cv:
        score *= 1.18

    # Short temperature history causes unstable CV; damp extreme overconfidence.
    if variable == "temp" and history_months < 24:
        if ms.model_key == "quant":
            score *= 1.18
        elif ms.model_key == "walkforward":
            score *= 1.30
        else:
            score *= 1.08
    elif variable == "temp" and history_months < 48 and ms.model_key == "walkforward":
        score *= 1.15

    # Sparse history should reduce confidence on high-variance members.
    if coverage_ratio < 0.45 and ms.model_key in {"quant", "strong"}:
        score *= 1.10
    if history_years < 5 and ms.model_key == "walkforward":
        score *= 1.12
    if recency_gap_years >= 3.0:
        score *= 1.0 + min(1.20, 0.06 * max(0.0, recency_gap_years - 2.0))
    if variable == "temp" and recency_gap_years >= 8.0 and ms.model_key in {"quant", "walkforward"}:
        score *= 1.22
    if ms.model_key == "climo_decay":
        score *= 0.96
        if recency_gap_years >= 8.0:
            score *= 0.82
        if history_months < 24:
            score *= 0.88
    if "__v_" in str(ms.model_key):
        # Virtual experts are useful but should remain regularized against over-dominance.
        score *= 1.06
    elif recency_gap_years >= 12.0 and ms.model_key in {"quant", "strong", "prophet_ultra", "walkforward"}:
        score *= 1.10

    return float(max(score, 1e-6))


def member_forecast_volatility_ratio(
    dfm: pd.DataFrame,
    forecast_start_year: int,
    target_year: int,
    ref_q95: float,
) -> float:
    if dfm is None or dfm.empty or "is_forecast" not in dfm.columns or "ds" not in dfm.columns or "yhat" not in dfm.columns:
        return float("nan")
    d = dfm[dfm["is_forecast"] == True].copy()
    if d.empty:
        return float("nan")
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d["yhat"] = pd.to_numeric(d["yhat"], errors="coerce")
    d = d.dropna(subset=["ds", "yhat"]).sort_values("ds")
    d = d[(d["ds"].dt.year >= int(forecast_start_year)) & (d["ds"].dt.year <= int(target_year))]
    if len(d) < 6:
        return float("nan")
    dd = np.abs(np.diff(d["yhat"].values.astype(float)))
    if dd.size == 0:
        return float("nan")
    q95 = float(np.quantile(dd, 0.95))
    return float(q95 / max(float(ref_q95), 1e-6))


def cap_and_normalize_weights(w: dict[str, float], min_w: float, max_w: float) -> dict[str, float]:
    if not w:
        return {}
    keys = list(w.keys())
    arr = np.array([float(w[k]) for k in keys], dtype=float)
    arr = np.clip(arr, 0.0, None)
    if not np.isfinite(arr).all() or arr.sum() <= 0:
        arr = np.ones_like(arr) / len(arr)
    else:
        arr = arr / arr.sum()

    lo = float(np.clip(min_w, 0.0, 1.0))
    hi = float(np.clip(max_w, lo, 1.0))
    arr = np.clip(arr, lo, hi)
    s = arr.sum()
    if s <= 0:
        arr = np.ones_like(arr) / len(arr)
    else:
        arr = arr / s
    return {k: float(v) for k, v in zip(keys, arr)}


def member_history_series(dfm: pd.DataFrame) -> pd.Series:
    if dfm is None or dfm.empty:
        return pd.Series(dtype=float)
    if "is_forecast" in dfm.columns:
        h = dfm[dfm["is_forecast"] == False].copy()
    else:
        h = dfm.copy()
    if h.empty or "ds" not in h.columns or "yhat" not in h.columns:
        return pd.Series(dtype=float)
    h["ds"] = pd.to_datetime(h["ds"], errors="coerce")
    h["yhat"] = pd.to_numeric(h["yhat"], errors="coerce")
    h = h.dropna(subset=["ds", "yhat"])
    if h.empty:
        return pd.Series(dtype=float)
    s = h.groupby("ds")["yhat"].mean().sort_index().astype(float)
    return s


def rmse_on_overlap(y_true: pd.Series, y_pred: pd.Series, min_points: int = 6) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    m = yt.notna() & yp.notna()
    n = int(m.sum())
    if n < min_points:
        return float("nan")
    e = yt[m].values.astype(float) - yp[m].values.astype(float)
    return float(np.sqrt(np.mean(e**2)))


def blend_on_index(index: pd.DatetimeIndex, history_map: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    if len(index) == 0 or not weights:
        return pd.Series(index=index, dtype=float)
    num = np.zeros(len(index), dtype=float)
    den = np.zeros(len(index), dtype=float)
    for k, wk in weights.items():
        if wk <= 0:
            continue
        s = history_map.get(k)
        if s is None or s.empty:
            continue
        p = pd.to_numeric(s.reindex(index), errors="coerce")
        ok = p.notna().values
        if not np.any(ok):
            continue
        vals = p.values.astype(float)
        num[ok] += float(wk) * vals[ok]
        den[ok] += float(wk)
    out = np.full(len(index), np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return pd.Series(out, index=index, dtype=float)


def normalize_weights_dict(w: dict[str, float]) -> dict[str, float]:
    if not w:
        return {}
    keys = list(w.keys())
    arr = np.array([max(0.0, float(w[k])) for k in keys], dtype=float)
    if not np.isfinite(arr).all() or arr.sum() <= 0:
        arr = np.ones(len(keys), dtype=float) / max(1, len(keys))
    else:
        arr = arr / arr.sum()
    return {k: float(v) for k, v in zip(keys, arr)}


def weights_from_train_rmse(
    subset: tuple[str, ...],
    train_idx: pd.DatetimeIndex,
    obs_series: pd.Series,
    history_map: dict[str, pd.Series],
    score_map: dict[str, float],
) -> dict[str, float]:
    rmse_map: dict[str, float] = {}
    for k in subset:
        s = history_map.get(k, pd.Series(dtype=float))
        rm = rmse_on_overlap(obs_series.reindex(train_idx), s.reindex(train_idx), min_points=6)
        rmse_map[k] = rm

    inv_rmse: dict[str, float] = {}
    for k in subset:
        rm = rmse_map.get(k, np.nan)
        if np.isfinite(rm) and rm > 1e-6:
            inv_rmse[k] = 1.0 / float(rm)
        else:
            sc = float(max(score_map.get(k, 1.0), 1e-6))
            inv_rmse[k] = 1.0 / sc
    return normalize_weights_dict(inv_rmse)


def evaluate_subset_rolling_cv(
    subset: tuple[str, ...],
    obs_series: pd.Series,
    history_map: dict[str, pd.Series],
    score_map: dict[str, float],
) -> tuple[float, int]:
    y = pd.to_numeric(obs_series, errors="coerce").dropna().sort_index()
    n = int(len(y))
    if n < 24:
        return float("nan"), 0

    n_folds = int(np.clip(n // 36, 2, 5))
    test_len = int(np.clip(n // (n_folds + 2), 6, 18))
    fold_rmses: list[float] = []

    for fi in range(n_folds):
        test_end = n - (n_folds - 1 - fi) * test_len
        test_start = test_end - test_len
        train_end = test_start
        if train_end < 12:
            continue
        train_idx = y.index[:train_end]
        test_idx = y.index[test_start:test_end]
        if len(test_idx) < 3:
            continue
        w = weights_from_train_rmse(
            subset=subset,
            train_idx=pd.DatetimeIndex(train_idx),
            obs_series=y,
            history_map=history_map,
            score_map=score_map,
        )
        pred = blend_on_index(pd.DatetimeIndex(test_idx), history_map=history_map, weights=w)
        rm = rmse_on_overlap(y.reindex(test_idx), pred, min_points=3)
        if np.isfinite(rm):
            fold_rmses.append(float(rm))

    if not fold_rmses:
        return float("nan"), 0
    return float(np.mean(fold_rmses)), int(len(fold_rmses))


def final_weights_for_subset(
    subset: tuple[str, ...],
    obs_series: pd.Series,
    history_map: dict[str, pd.Series],
    score_map: dict[str, float],
) -> dict[str, float]:
    y = pd.to_numeric(obs_series, errors="coerce").dropna().sort_index()
    inv_hist: dict[str, float] = {}
    hist_ok = 0
    for k in subset:
        s = history_map.get(k, pd.Series(dtype=float))
        rm = rmse_on_overlap(y, s.reindex(y.index), min_points=8)
        if np.isfinite(rm) and rm > 1e-6:
            inv_hist[k] = 1.0 / float(rm)
            hist_ok += 1
        else:
            inv_hist[k] = 0.0

    inv_score: dict[str, float] = {}
    for k in subset:
        sc = float(max(score_map.get(k, 1.0), 1e-6))
        inv_score[k] = 1.0 / sc

    w_hist = normalize_weights_dict(inv_hist)
    w_score = normalize_weights_dict(inv_score)
    if hist_ok >= 2:
        hist_mix = float(np.clip(0.45 + (len(y) / 240.0), 0.45, 0.78))
    else:
        hist_mix = 0.0

    out: dict[str, float] = {}
    for k in subset:
        out[k] = hist_mix * float(w_hist.get(k, 0.0)) + (1.0 - hist_mix) * float(w_score.get(k, 0.0))
    return normalize_weights_dict(out)


def metric_combo_selection(
    score_df: pd.DataFrame,
    max_combo_size: int,
    complexity_penalty: float,
    obs_scale: float,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, Any]]:
    keys = list(dict.fromkeys(score_df["model_key"].astype(str).tolist()))
    if not keys:
        return {}, pd.DataFrame(), {"reason": "no_models"}

    rmse_med = float(pd.to_numeric(score_df["rmse"], errors="coerce").median())
    if not np.isfinite(rmse_med):
        rmse_med = max(0.5, 0.10 * float(max(obs_scale, 1.0)))

    kmax = int(np.clip(max_combo_size, 1, len(keys)))
    rows: list[dict[str, Any]] = []
    for sz in range(1, kmax + 1):
        for combo in itertools.combinations(keys, sz):
            sub = score_df[score_df["model_key"].astype(str).isin(combo)].copy()
            if sub.empty:
                continue
            inv = 1.0 / np.maximum(pd.to_numeric(sub["score"], errors="coerce").fillna(rmse_med).values.astype(float), 1e-6)
            ww = inv / np.sum(inv)
            w = {str(k): float(v) for k, v in zip(sub["model_key"].astype(str).tolist(), ww)}
            w = normalize_weights_dict(w)

            rm_mix = 0.0
            b_mix = 0.0
            s_mix = 0.0
            signs: list[float] = []
            for _, r in sub.iterrows():
                k = str(r["model_key"])
                wk = float(w.get(k, 0.0))
                rm = safe_float(r.get("rmse"), default=rmse_med)
                if not np.isfinite(rm):
                    rm = rmse_med
                bi = safe_float(r.get("bias"), default=0.0)
                st = abs(safe_float(r.get("rmse_std"), default=0.0))
                rm_mix += wk * float(rm)
                b_mix += wk * float(bi)
                s_mix += wk * float(st)
                signs.append(float(np.sign(bi)))

            rm_mix = float(max(rm_mix, 1e-6))
            bias_norm = abs(float(b_mix)) / max(1.0, 0.08 * float(max(obs_scale, 1.0)))
            stab_norm = float(s_mix) / rm_mix
            surrogate = rm_mix * (1.0 + 0.22 * bias_norm + 0.12 * stab_norm)
            if sz >= 2 and any(v > 0 for v in signs) and any(v < 0 for v in signs):
                surrogate *= 0.97
            penalized = surrogate * (1.0 + float(max(0.0, complexity_penalty)) * (sz - 1))

            rows.append(
                {
                    "models": ",".join(combo),
                    "model_count": int(sz),
                    "cv_rmse": np.nan,
                    "cv_folds": 0,
                    "history_rmse": np.nan,
                    "surrogate_obj": float(surrogate),
                    "penalized_cv": float(penalized),
                    "weights_json": json.dumps(w, ensure_ascii=False),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return {}, table, {"reason": "metric_combo_failed"}
    table = table.sort_values(["penalized_cv", "model_count"], ascending=[True, True]).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    best_weights = json.loads(str(best["weights_json"]))
    return (
        {str(k): float(v) for k, v in best_weights.items()},
        table,
        {
            "reason": "metric_surrogate",
            "selection_method": "metric_surrogate",
            "selected_models": str(best["models"]).split(","),
            "selected_model_count": int(best["model_count"]),
            "selected_cv_rmse": None,
            "selected_penalized_cv": float(best["penalized_cv"]),
            "selected_surrogate_obj": float(best["surrogate_obj"]),
        },
    )


def auto_select_combination(
    score_df: pd.DataFrame,
    obs_series: pd.Series,
    members: list[tuple[ModelScore, pd.DataFrame]],
    max_combo_size: int,
    complexity_penalty: float,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, Any]]:
    history_map: dict[str, pd.Series] = {}
    for ms, dfm in members:
        if ms.model_key not in history_map:
            history_map[ms.model_key] = member_history_series(dfm)

    score_map = {
        str(r["model_key"]): float(max(float(r["score"]), 1e-6))
        for _, r in score_df.iterrows()
        if pd.notna(r.get("score"))
    }
    keys = [k for k in score_df["model_key"].astype(str).tolist() if k in history_map]
    keys = list(dict.fromkeys(keys))
    if not keys:
        return {}, pd.DataFrame(), {"reason": "no_history_for_auto_selection"}

    obs_clean = pd.to_numeric(obs_series, errors="coerce").dropna().sort_index()
    obs_scale = float(np.nanmedian(np.abs(obs_clean.values))) if len(obs_clean) else 1.0
    obs_scale = float(max(obs_scale, 1.0))
    trust_threshold = float(max(0.75, 0.03 * obs_scale))
    trusted_keys: list[str] = []
    for k in keys:
        hs = history_map.get(k, pd.Series(dtype=float))
        rm = rmse_on_overlap(obs_clean, hs.reindex(obs_clean.index), min_points=8)
        # Very tiny in-sample error often means target leakage (history=yhat copy).
        if np.isfinite(rm) and rm >= trust_threshold:
            trusted_keys.append(k)

    kmax = int(np.clip(max_combo_size, 1, len(keys)))
    rows: list[dict[str, Any]] = []

    if len(trusted_keys) >= 2 and len(obs_clean) >= 24:
        cv_keys = trusted_keys
        kmax_cv = int(np.clip(max_combo_size, 1, len(cv_keys)))
    else:
        cv_keys = []
        kmax_cv = 0

    for sz in range(1, kmax_cv + 1):
        for combo in itertools.combinations(cv_keys, sz):
            cv_rmse, folds = evaluate_subset_rolling_cv(
                subset=combo,
                obs_series=obs_series,
                history_map=history_map,
                score_map=score_map,
            )
            if not np.isfinite(cv_rmse):
                continue
            w_final = final_weights_for_subset(
                subset=combo,
                obs_series=obs_series,
                history_map=history_map,
                score_map=score_map,
            )
            pred_all = blend_on_index(pd.DatetimeIndex(obs_series.dropna().sort_index().index), history_map=history_map, weights=w_final)
            hist_rmse = rmse_on_overlap(obs_series, pred_all, min_points=8)
            pen = float(cv_rmse) * (1.0 + float(max(0.0, complexity_penalty)) * (sz - 1))
            rows.append(
                {
                    "models": ",".join(combo),
                    "model_count": int(sz),
                    "cv_rmse": float(cv_rmse),
                    "cv_folds": int(folds),
                    "history_rmse": float(hist_rmse) if np.isfinite(hist_rmse) else np.nan,
                    "penalized_cv": float(pen),
                    "weights_json": json.dumps(w_final, ensure_ascii=False),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        w2, t2, m2 = metric_combo_selection(
            score_df=score_df,
            max_combo_size=int(max_combo_size),
            complexity_penalty=float(complexity_penalty),
            obs_scale=float(obs_scale),
        )
        if isinstance(m2, dict):
            m2["trusted_history_models"] = trusted_keys
            m2["trust_rmse_threshold"] = trust_threshold
        return w2, t2, m2

    table = table.sort_values(["penalized_cv", "model_count", "cv_rmse"], ascending=[True, True, True]).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    best_weights = json.loads(str(best["weights_json"]))
    return (
        {str(k): float(v) for k, v in best_weights.items()},
        table,
        {
            "reason": "ok",
            "selection_method": "rolling_cv_history",
            "trusted_history_models": trusted_keys,
            "trust_rmse_threshold": trust_threshold,
            "selected_models": str(best["models"]).split(","),
            "selected_model_count": int(best["model_count"]),
            "selected_cv_rmse": float(best["cv_rmse"]),
            "selected_penalized_cv": float(best["penalized_cv"]),
        },
    )


def horizon_weight_multiplier(
    model_key: str,
    horizon01: np.ndarray,
    variable: str,
    coverage_ratio: float,
    recency_gap_years: float,
) -> np.ndarray:
    h = np.asarray(horizon01, dtype=float)
    if h.size == 0:
        return np.array([], dtype=float)
    h = np.clip(h, 0.0, 1.0)
    key = str(model_key)
    stale = float(np.clip((float(recency_gap_years) - 2.0) / 10.0, 0.0, 1.0))
    sparse = float(np.clip((0.60 - float(coverage_ratio)) / 0.35, 0.0, 1.0))

    var_scale = 1.0
    if variable == "temp":
        var_scale = 1.12
    elif variable == "precip":
        var_scale = 0.92
    elif variable == "pressure":
        var_scale = 0.96

    if key == "climo_decay":
        start, end = 1.06, 1.58 + 0.14 * stale + 0.08 * sparse
    elif key.endswith("__v_damped"):
        start, end = 1.03, 1.44 + 0.12 * stale + 0.06 * sparse
    elif key.endswith("__v_climo"):
        start, end = 1.02, 1.34 + 0.10 * stale + 0.05 * sparse
    elif key.endswith("__v_smooth"):
        start, end = 1.01, 1.20 + 0.06 * sparse
    elif "__v_" in key:
        start, end = 1.00, 1.16 + 0.06 * sparse
    else:
        # Let raw members dominate short horizon but fade slightly with lead time.
        start, end = 1.00, 0.84 - 0.08 * stale - 0.06 * sparse

    start_adj = 1.0 + (float(start) - 1.0) * var_scale
    end_adj = 1.0 + (float(end) - 1.0) * var_scale
    mult = start_adj + (end_adj - start_adj) * h
    return np.clip(mult, 0.45, 2.05)


def weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) == 0:
        return float("nan")
    vv = v[ok]
    ww = w[ok]
    if vv.size == 1:
        return float(vv[0])
    order = np.argsort(vv)
    vv = vv[order]
    ww = ww[order]
    cdf = np.cumsum(ww)
    tot = float(cdf[-1])
    if tot <= 0:
        return float(np.nanmedian(vv))
    tq = float(np.clip(q, 0.0, 1.0)) * tot
    idx = int(np.searchsorted(cdf, tq, side="left"))
    idx = int(np.clip(idx, 0, len(vv) - 1))
    return float(vv[idx])


def robust_reweight_1d(
    values: np.ndarray,
    weights: np.ndarray,
    variable: str,
    obs_scale: float,
) -> tuple[np.ndarray, float, float, float]:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = np.zeros_like(w, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return out, float("nan"), 0.0, 0.0
    vv = v[ok]
    ww = w[ok]
    if n_ok <= 2:
        out[ok] = ww
        center = float(np.dot(ww, vv) / max(np.sum(ww), 1e-12))
        return out, center, 0.0, 0.0

    med = weighted_quantile_1d(vv, ww, 0.5)
    dev = np.abs(vv - med)
    mad = weighted_quantile_1d(dev, ww, 0.5)

    if variable == "temp":
        floor = max(0.12, 0.018 * max(float(obs_scale), 1.0))
        k = 2.5
    elif variable == "humidity":
        floor = max(0.80, 0.022 * max(float(obs_scale), 1.0))
        k = 2.8
    elif variable == "precip":
        floor = max(1.80, 0.035 * max(float(obs_scale), 1.0))
        k = 3.4
    else:
        floor = max(0.50, 0.020 * max(float(obs_scale), 1.0))
        k = 2.8

    scale = float(max(1.4826 * mad if np.isfinite(mad) else 0.0, floor, 1e-6))
    z = dev / max(scale, 1e-9)
    # Smoothly down-weight outlier members without hard drops.
    robust = 1.0 / (1.0 + (z / max(k, 1e-6)) ** 2)
    robust = np.where(z > (4.0 * k), robust * 0.20, robust)
    ww2 = ww * robust
    if (not np.isfinite(ww2).all()) or float(np.sum(ww2)) <= 1e-12:
        ww2 = ww
    out[ok] = ww2

    raw_sum = float(np.sum(ww))
    eff_sum = float(np.sum(ww2))
    center = float(np.dot(ww2, vv) / max(eff_sum, 1e-12))
    suppression = float(np.clip(1.0 - eff_sum / max(raw_sum, 1e-12), 0.0, 1.0))
    return out, center, scale, suppression


def blend_variable(
    variable: str,
    obs_monthly: pd.Series,
    members: list[tuple[ModelScore, pd.DataFrame]],
    forecast_start_year: int,
    target_year: int,
    interval_z: float,
    min_w: float,
    max_w: float,
    auto_select_combination_flag: bool = True,
    max_combo_size: int = 3,
    combo_complexity_penalty: float = 0.025,
    force_combine_count: int = 0,
    min_temp_warming_rate_c_per_year: float = 0.0,
    continuity_alpha_override: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    obs_series = pd.to_numeric(obs_monthly, errors="coerce")
    ref_end = pd.Timestamp(year=max(int(forecast_start_year) - 1, 1), month=12, day=1)
    cov = monthly_history_coverage(obs_series, reference_end=ref_end)
    obs_valid = obs_series.dropna().sort_index()
    hist_months = int(cov["months_with_data"])
    hist_years = int(cov["years_with_data"])
    coverage_ratio = float(cov["coverage_ratio"])
    recency_gap_years = float(cov.get("recency_gap_years", 0.0))
    if hist_months <= 0:
        return pd.DataFrame(), {"weights": {}, "score_table": [], "history_coverage": cov}, pd.DataFrame()

    obs_scale = float(np.nanmedian(np.abs(obs_valid.values))) if len(obs_valid) else 1.0
    obs_scale = float(max(obs_scale, 1.0))
    obs_diff = pd.to_numeric(obs_valid, errors="coerce").dropna().diff().abs().dropna()
    if len(obs_diff) >= 4:
        obs_diff_q95 = float(obs_diff.quantile(0.95))
    else:
        obs_diff_q95 = float(0.10 * obs_scale)
    obs_diff_q95 = float(max(obs_diff_q95, 1e-6))

    score_rows: list[dict[str, Any]] = []
    for ms, dfm in members:
        score = model_score_value(
            ms,
            obs_scale=obs_scale,
            variable=variable,
            history_months=hist_months,
            history_years=hist_years,
            coverage_ratio=coverage_ratio,
            recency_gap_years=recency_gap_years,
        )
        vol_ratio = member_forecast_volatility_ratio(
            dfm=dfm,
            forecast_start_year=int(forecast_start_year),
            target_year=int(target_year),
            ref_q95=float(obs_diff_q95),
        )
        vol_penalty = 1.0
        if np.isfinite(vol_ratio):
            if variable == "temp":
                allow, lam, reward_thr = 2.1, 0.11, 0.72
            elif variable == "humidity":
                allow, lam, reward_thr = 2.8, 0.08, 0.70
            elif variable == "precip":
                allow, lam, reward_thr = 3.9, 0.06, 0.65
            else:
                allow, lam, reward_thr = 3.0, 0.08, 0.70
            if vol_ratio > allow:
                vol_penalty *= float(min(1.55, 1.0 + lam * (vol_ratio - allow)))
            elif vol_ratio < reward_thr:
                vol_penalty *= float(max(0.90, 1.0 - 0.04 * (reward_thr - vol_ratio)))
        score *= float(vol_penalty)
        score_rows.append(
            {
                "model_key": ms.model_key,
                "variable": variable,
                "frequency": ms.frequency,
                "rmse": float(ms.rmse) if np.isfinite(ms.rmse) else np.nan,
                "bias": float(ms.bias),
                "rmse_std": float(ms.rmse_std),
                "has_cv": bool(ms.has_cv),
                "score": float(score),
                "volatility_ratio_q95": float(vol_ratio) if np.isfinite(vol_ratio) else np.nan,
                "volatility_penalty": float(vol_penalty),
                "forecast_csv": ms.forecast_csv,
                "source_index": ms.source,
            }
        )

    valid_members: list[tuple[ModelScore, pd.DataFrame]] = []
    invalid_keys: list[str] = []
    for ms, dfm in members:
        if member_is_plausible(
            variable=variable,
            obs_monthly=obs_monthly,
            dfm=dfm,
            forecast_start_year=int(forecast_start_year),
            target_year=int(target_year),
        ):
            valid_members.append((ms, dfm))
        else:
            invalid_keys.append(ms.model_key)

    if valid_members:
        members = valid_members

    score_df = pd.DataFrame(score_rows).sort_values("score", ascending=True).reset_index(drop=True)
    if invalid_keys:
        score_df["plausible_member"] = ~score_df["model_key"].isin(invalid_keys)
        score_df = score_df[score_df["plausible_member"] == True].copy().reset_index(drop=True)
    else:
        score_df["plausible_member"] = True
    if score_df.empty:
        return (
            pd.DataFrame(),
            {"weights": {}, "score_table": [], "invalid_members": invalid_keys, "history_coverage": cov},
            pd.DataFrame(),
        )

    inv = 1.0 / np.maximum(score_df["score"].values.astype(float), 1e-6)
    w_raw = inv / np.sum(inv)
    score_w = {str(k): float(v) for k, v in zip(score_df["model_key"].tolist(), w_raw)}
    combo_table = pd.DataFrame()
    combo_meta: dict[str, Any] = {"reason": "disabled_or_single_model"}

    if int(force_combine_count) > 1 and len(score_df) >= 2:
        keep_n = int(np.clip(int(force_combine_count), 2, len(score_df)))
        keep_keys = score_df.head(keep_n)["model_key"].astype(str).tolist()
        members = [(ms, dfm) for ms, dfm in members if ms.model_key in set(keep_keys)]
        score_df["selected_in_forced_combo"] = score_df["model_key"].astype(str).isin(set(keep_keys))
        score_df = score_df[score_df["selected_in_forced_combo"] == True].copy().reset_index(drop=True)
        inv_forced = 1.0 / np.maximum(score_df["score"].values.astype(float), 1e-6)
        w_forced = inv_forced / np.sum(inv_forced)
        w_forced_map = {str(k): float(v) for k, v in zip(score_df["model_key"].tolist(), w_forced)}
        w = cap_and_normalize_weights(w_forced_map, min_w=min_w, max_w=max_w)
        combo_meta = {
            "reason": "forced_combine_count",
            "selection_method": "forced_top_n_by_score",
            "selected_models": keep_keys,
            "selected_model_count": int(len(keep_keys)),
            "selected_cv_rmse": None,
            "selected_penalized_cv": None,
        }
    elif auto_select_combination_flag and len(score_df) >= 2:
        auto_w, combo_table, combo_meta = auto_select_combination(
            score_df=score_df,
            obs_series=obs_series,
            members=members,
            max_combo_size=int(max_combo_size),
            complexity_penalty=float(combo_complexity_penalty),
        )
        if auto_w:
            selected_keys = set(auto_w.keys())
            members = [(ms, dfm) for ms, dfm in members if ms.model_key in selected_keys]
            score_df["selected_in_best_combo"] = score_df["model_key"].isin(selected_keys)
            score_df = score_df[score_df["selected_in_best_combo"] == True].copy().reset_index(drop=True)
            w = cap_and_normalize_weights(auto_w, min_w=min_w, max_w=max_w)
        else:
            w = cap_and_normalize_weights(score_w, min_w=min_w, max_w=max_w)
    else:
        w = cap_and_normalize_weights(score_w, min_w=min_w, max_w=max_w)

    score_df["weight"] = score_df["model_key"].map(w).fillna(0.0)

    # Build forecast index (monthly horizon up to target year).
    all_future = []
    for _, dfm in members:
        fc = dfm[dfm["is_forecast"] == True][["ds"]].copy()
        all_future.append(fc)
    future_idx = pd.concat(all_future, ignore_index=True).drop_duplicates().sort_values("ds")
    future_idx = future_idx[future_idx["ds"].dt.year >= int(forecast_start_year)]
    future_idx = future_idx[future_idx["ds"].dt.year <= int(target_year)]
    if future_idx.empty:
        return pd.DataFrame(), {"weights": w, "score_table": score_df.to_dict(orient="records"), "history_coverage": cov}, score_df

    blend = future_idx.copy().reset_index(drop=True)
    blend["yhat"] = np.nan
    blend["yhat_lower"] = np.nan
    blend["yhat_upper"] = np.nan
    blend["w_sum"] = 0.0
    blend["w_lo_sum"] = 0.0
    blend["w_hi_sum"] = 0.0
    blend["weighted_climate_delta"] = 0.0

    if len(blend) <= 1:
        horizon01 = np.zeros(len(blend), dtype=float)
    else:
        h_month = (
            (blend["ds"].dt.year.astype(int) - int(forecast_start_year)) * 12
            + (blend["ds"].dt.month.astype(int) - 1)
        ).values.astype(float)
        h_month = h_month - float(np.nanmin(h_month))
        h_span = float(np.nanmax(h_month))
        if np.isfinite(h_span) and h_span > 0:
            horizon01 = np.clip(h_month / h_span, 0.0, 1.0)
        else:
            horizon01 = np.linspace(0.0, 1.0, len(blend), dtype=float)

    month_clim_anchor = smooth_month_climatology(obs_valid, variable=variable) if hist_months >= 12 else {}
    if month_clim_anchor:
        blend_month_anchor = np.array([float(month_clim_anchor.get(int(ts.month), np.nan)) for ts in blend["ds"]], dtype=float)
    else:
        blend_month_anchor = np.full(len(blend), np.nan, dtype=float)

    model_rmse = {str(r["model_key"]): safe_float(r["rmse"], default=np.nan) for _, r in score_df.iterrows()}
    effective_weight_mean: dict[str, float] = {}
    horizon_weight_profiles: list[dict[str, Any]] = []
    pseudo_delta_models: list[str] = []
    pred_stack: list[np.ndarray] = []
    lo_stack: list[np.ndarray] = []
    hi_stack: list[np.ndarray] = []
    cdel_stack: list[np.ndarray] = []
    weight_stack: list[np.ndarray] = []
    robust_enabled = variable in {"humidity"}
    robust_mode = "huber" if robust_enabled else "disabled_for_variable"

    for ms, dfm in members:
        key = ms.model_key
        wk = float(w.get(key, 0.0))
        if wk <= 0:
            continue
        fc = dfm[dfm["is_forecast"] == True].copy()
        if fc.empty:
            continue
        fc = fc[fc["ds"].dt.year >= int(forecast_start_year)].copy()
        fc = fc[fc["ds"].dt.year <= int(target_year)].copy()
        if fc.empty:
            continue
        fc = fc.set_index("ds").sort_index()
        wk_vec = float(wk) * horizon_weight_multiplier(
            model_key=key,
            horizon01=horizon01,
            variable=variable,
            coverage_ratio=coverage_ratio,
            recency_gap_years=recency_gap_years,
        )
        if wk_vec.shape[0] != len(blend):
            wk_vec = np.full(len(blend), float(wk), dtype=float)
        wk_vec = np.clip(wk_vec, 0.0, None).astype(float)
        if not np.any(wk_vec > 0):
            continue
        effective_weight_mean[key] = float(np.nanmean(wk_vec))
        if len(wk_vec) > 0:
            horizon_weight_profiles.append(
                {
                    "model_key": str(key),
                    "base_weight": float(wk),
                    "weight_start": float(wk_vec[0]),
                    "weight_mid": float(wk_vec[len(wk_vec) // 2]),
                    "weight_end": float(wk_vec[-1]),
                    "multiplier_start": float(wk_vec[0] / max(float(wk), 1e-12)),
                    "multiplier_end": float(wk_vec[-1] / max(float(wk), 1e-12)),
                }
            )

        yhat = fc["yhat"].reindex(blend["ds"]).astype(float)
        ylo = fc["yhat_lower"].reindex(blend["ds"]).astype(float)
        yhi = fc["yhat_upper"].reindex(blend["ds"]).astype(float)
        cdel = fc["climate_delta"].reindex(blend["ds"]).astype(float) if "climate_delta" in fc.columns else pd.Series(0.0, index=blend["ds"])
        if variable in {"temp", "humidity"} and recency_gap_years >= 2.0:
            cdel_abs_mean = float(np.nanmean(np.abs(pd.to_numeric(cdel, errors="coerce").fillna(0.0).values)))
            if cdel_abs_mean <= 1e-6 and np.isfinite(blend_month_anchor).sum() >= max(6, len(blend) // 3):
                yhat_vals = yhat.values.astype(float)
                pseudo = np.full(len(blend), np.nan, dtype=float)
                ok = np.isfinite(yhat_vals) & np.isfinite(blend_month_anchor)
                pseudo[ok] = yhat_vals[ok] - blend_month_anchor[ok]
                pser = pd.Series(pseudo, index=blend["ds"], dtype=float)
                pser = pser.rolling(window=12, min_periods=3, center=True).mean()
                pser = pser.interpolate(limit_direction="both").fillna(0.0)
                cdel = pser
                pseudo_delta_models.append(str(key))

        # Interval fallbacks if a member does not provide bounds.
        rm = model_rmse.get(key, np.nan)
        if not np.isfinite(rm):
            rm = max(0.5, 0.10 * obs_scale)
        ylo_f = ylo.copy()
        yhi_f = yhi.copy()
        miss_lo = ~ylo_f.notna()
        miss_hi = ~yhi_f.notna()
        ylo_f.loc[miss_lo & yhat.notna()] = (yhat - interval_z * rm).loc[miss_lo & yhat.notna()]
        yhi_f.loc[miss_hi & yhat.notna()] = (yhat + interval_z * rm).loc[miss_hi & yhat.notna()]

        pred_stack.append(yhat.values.astype(float))
        lo_stack.append(ylo_f.values.astype(float))
        hi_stack.append(yhi_f.values.astype(float))
        cdel_stack.append(pd.to_numeric(cdel, errors="coerce").values.astype(float))
        weight_stack.append(wk_vec.astype(float))

    if not pred_stack:
        return pd.DataFrame(), {"weights": w, "score_table": score_df.to_dict(orient="records"), "history_coverage": cov}, score_df

    pred_mat = np.vstack(pred_stack)
    lo_mat = np.vstack(lo_stack)
    hi_mat = np.vstack(hi_stack)
    cdel_mat = np.vstack(cdel_stack)
    w_mat = np.vstack(weight_stack)

    n_steps = len(blend)
    yhat_out = np.full(n_steps, np.nan, dtype=float)
    ylo_out = np.full(n_steps, np.nan, dtype=float)
    yhi_out = np.full(n_steps, np.nan, dtype=float)
    cdel_out = np.zeros(n_steps, dtype=float)
    w_sum_arr = np.zeros(n_steps, dtype=float)
    w_lo_sum_arr = np.zeros(n_steps, dtype=float)
    w_hi_sum_arr = np.zeros(n_steps, dtype=float)
    robust_scale_arr = np.zeros(n_steps, dtype=float)
    robust_supp_arr = np.zeros(n_steps, dtype=float)

    for j in range(n_steps):
        if robust_enabled:
            adj_w, center, scale, suppress = robust_reweight_1d(
                values=pred_mat[:, j],
                weights=w_mat[:, j],
                variable=variable,
                obs_scale=float(obs_scale),
            )
        else:
            pcol = pred_mat[:, j]
            wcol = w_mat[:, j]
            ok_base = np.isfinite(pcol) & np.isfinite(wcol) & (wcol > 0)
            adj_w = np.where(ok_base, wcol, 0.0).astype(float)
            sw = float(np.sum(adj_w))
            center = float(np.dot(adj_w[ok_base], pcol[ok_base]) / max(sw, 1e-12)) if sw > 0 else float("nan")
            scale = 0.0
            suppress = 0.0
        sum_w = float(np.sum(adj_w))
        if sum_w <= 0 or not np.isfinite(center):
            continue
        yhat_out[j] = float(center)
        w_sum_arr[j] = float(sum_w)
        robust_scale_arr[j] = float(scale)
        robust_supp_arr[j] = float(suppress)

        ccol = cdel_mat[:, j]
        okc = np.isfinite(ccol) & (adj_w > 0)
        if np.any(okc):
            wc = adj_w[okc]
            cdel_out[j] = float(np.dot(wc, ccol[okc]) / max(np.sum(wc), 1e-12))

        locol = lo_mat[:, j]
        ok_lo = np.isfinite(locol) & (adj_w > 0)
        if np.any(ok_lo):
            wlo = adj_w[ok_lo]
            ylo_out[j] = float(np.dot(wlo, locol[ok_lo]) / max(np.sum(wlo), 1e-12))
            w_lo_sum_arr[j] = float(np.sum(wlo))

        hicol = hi_mat[:, j]
        ok_hi = np.isfinite(hicol) & (adj_w > 0)
        if np.any(ok_hi):
            whi = adj_w[ok_hi]
            yhi_out[j] = float(np.dot(whi, hicol[ok_hi]) / max(np.sum(whi), 1e-12))
            w_hi_sum_arr[j] = float(np.sum(whi))

    blend["yhat"] = yhat_out
    blend["yhat_lower"] = ylo_out
    blend["yhat_upper"] = yhi_out
    blend["weighted_climate_delta"] = cdel_out
    blend["w_sum"] = w_sum_arr
    blend["w_lo_sum"] = w_lo_sum_arr
    blend["w_hi_sum"] = w_hi_sum_arr

    robust_ok = np.isfinite(yhat_out) & (w_sum_arr > 0)
    if np.any(robust_ok):
        sup_vals = robust_supp_arr[robust_ok]
        sc_vals = robust_scale_arr[robust_ok]
        robust_blend_stats = {
            "enabled": bool(robust_enabled),
            "mode": str(robust_mode),
            "mean_suppression": float(np.mean(sup_vals)),
            "p90_suppression": float(np.quantile(sup_vals, 0.90)),
            "max_suppression": float(np.max(sup_vals)),
            "mean_scale": float(np.mean(sc_vals)),
            "p90_scale": float(np.quantile(sc_vals, 0.90)),
        }
    else:
        robust_blend_stats = {
            "enabled": bool(robust_enabled),
            "mode": str(robust_mode),
            "mean_suppression": 0.0,
            "p90_suppression": 0.0,
            "max_suppression": 0.0,
            "mean_scale": 0.0,
            "p90_scale": 0.0,
        }

    ok = blend["w_sum"] > 0
    blend = blend[ok].copy().reset_index(drop=True)
    if blend.empty:
        return pd.DataFrame(), {"weights": w, "score_table": score_df.to_dict(orient="records"), "history_coverage": cov}, score_df

    # If still missing intervals, create from weighted RMSE proxy.
    rmse_proxy = 0.0
    rmse_weight_map = normalize_weights_dict(effective_weight_mean) if effective_weight_mean else normalize_weights_dict(w)
    for k, wk in rmse_weight_map.items():
        rm = model_rmse.get(k, np.nan)
        if not np.isfinite(rm):
            rm = max(0.5, 0.10 * obs_scale)
        rmse_proxy += float(wk) * float(rm)
    rmse_proxy = float(max(rmse_proxy, 1e-6))

    miss_lo = ~blend["yhat_lower"].notna()
    miss_hi = ~blend["yhat_upper"].notna()
    blend.loc[miss_lo, "yhat_lower"] = blend.loc[miss_lo, "yhat"] - interval_z * rmse_proxy
    blend.loc[miss_hi, "yhat_upper"] = blend.loc[miss_hi, "yhat"] + interval_z * rmse_proxy

    # Physics-aware anchoring to monthly climatology + climate drift.
    mean_rev = np.zeros(len(blend), dtype=float)
    if hist_months >= 12 and int(cov["month_of_year_coverage"]) >= 8:
        month_clim = smooth_month_climatology(obs_valid, variable=variable)
        if len(month_clim) < 8:
            month_clim = obs_valid.groupby(obs_valid.index.month).mean().to_dict()
        n = len(blend)
        for i, ts in enumerate(blend["ds"].tolist(), start=1):
            ridx = blend.index[i - 1]
            m = int(pd.Timestamp(ts).month)
            if m not in month_clim:
                continue
            anchor = float(month_clim[m]) + float(blend.at[ridx, "weighted_climate_delta"])
            if variable == "temp":
                if hist_months < 24:
                    w0, w1 = 0.45, 0.75
                elif hist_months < 60:
                    w0, w1 = 0.25, 0.55
                else:
                    w0, w1 = 0.15, 0.35
            elif variable == "humidity":
                w0, w1 = 0.20, 0.42
            else:
                w0, w1 = 0.14, 0.30
            rev_w = float(np.clip(w0 + (w1 - w0) * (i / max(n, 1)), w0, w1))
            y_prev = float(blend.at[ridx, "yhat"])
            y_new = (1.0 - rev_w) * y_prev + rev_w * anchor
            shift = y_new - y_prev
            blend.at[ridx, "yhat"] = y_new
            blend.at[ridx, "yhat_lower"] = float(blend.at[ridx, "yhat_lower"]) + shift
            blend.at[ridx, "yhat_upper"] = float(blend.at[ridx, "yhat_upper"]) + shift
            mean_rev[i - 1] = float(shift)
    blend["mean_reversion"] = mean_rev

    # Short-history inflation for interval realism.
    interval_inflation = 1.0
    if hist_months < 24:
        interval_inflation *= 1.35
    elif hist_months < 60:
        interval_inflation *= 1.12
    if coverage_ratio < 0.40:
        interval_inflation *= 1.20
    elif coverage_ratio < 0.65:
        interval_inflation *= 1.08
    if int(cov["month_of_year_coverage"]) < 8:
        interval_inflation *= 1.12

    if interval_inflation > 1.001 and len(blend):
        half = (blend["yhat_upper"].values.astype(float) - blend["yhat_lower"].values.astype(float)) / 2.0
        half = np.maximum(half, 1e-6) * float(interval_inflation)
        center = blend["yhat"].values.astype(float)
        blend["yhat_lower"] = center - half
        blend["yhat_upper"] = center + half

    blend, consistency_stats = apply_consistency_regularization(
        blend=blend,
        obs_valid=obs_valid,
        variable=variable,
        coverage=cov,
    )
    blend, continuity_stats = apply_continuity_smoothing(
        blend=blend,
        variable=variable,
        coverage=cov,
        alpha_override=continuity_alpha_override,
    )

    trend_adjust_mean = 0.0
    trend_adjust_max = 0.0
    if variable == "temp" and float(min_temp_warming_rate_c_per_year) > 0 and len(blend) >= 24:
        ann = (
            blend.assign(year=pd.to_datetime(blend["ds"]).dt.year)
            .groupby("year", as_index=False)["yhat"]
            .mean()
            .dropna()
            .sort_values("year")
        )
        if len(ann) >= 2:
            years_span = float(max(int(ann["year"].iloc[-1]) - int(ann["year"].iloc[0]), 1))
            slope = float((ann["yhat"].iloc[-1] - ann["yhat"].iloc[0]) / years_span)
            min_rate = float(max(0.0, min_temp_warming_rate_c_per_year))
            if slope < min_rate:
                need = min_rate - slope
                t0 = pd.Timestamp(blend["ds"].min())
                years_from_start = (pd.to_datetime(blend["ds"]) - t0).dt.total_seconds().values.astype(float) / (
                    365.25 * 24.0 * 3600.0
                )
                shift = need * years_from_start
                blend["yhat"] = blend["yhat"].values.astype(float) + shift
                blend["yhat_lower"] = blend["yhat_lower"].values.astype(float) + shift
                blend["yhat_upper"] = blend["yhat_upper"].values.astype(float) + shift
                trend_adjust_mean = float(np.mean(shift))
                trend_adjust_max = float(np.max(shift))

    y = apply_bounds(blend["yhat"].values.astype(float), variable)
    lo = apply_bounds(blend["yhat_lower"].values.astype(float), variable)
    hi = apply_bounds(blend["yhat_upper"].values.astype(float), variable)
    lo = np.minimum(lo, y)
    hi = np.maximum(hi, y)
    blend["yhat"] = y
    blend["yhat_lower"] = lo
    blend["yhat_upper"] = hi

    hist = pd.DataFrame(
        {
            "ds": pd.to_datetime(obs_valid.index),
            "actual": obs_valid.values.astype(float),
            "yhat": obs_valid.values.astype(float),
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
            "is_forecast": False,
            "mean_reversion": 0.0,
            "climate_delta": 0.0,
        }
    )
    fc = pd.DataFrame(
        {
            "ds": blend["ds"].values,
            "actual": np.nan,
            "yhat": blend["yhat"].values,
            "yhat_lower": blend["yhat_lower"].values,
            "yhat_upper": blend["yhat_upper"].values,
            "is_forecast": True,
            "mean_reversion": blend["mean_reversion"].values,
            "climate_delta": blend["weighted_climate_delta"].values,
        }
    )
    out_df = pd.concat([hist, fc], ignore_index=True)
    out_df["variable"] = variable
    out_df["unit"] = infer_unit(variable)
    out_df["frequency"] = "MS"
    out_df["model_strategy"] = "best_meta_ensemble"
    return (
        out_df,
        {
            "weights": w,
            "score_table": score_df.to_dict(orient="records"),
            "rmse_proxy": rmse_proxy,
            "invalid_members": invalid_keys,
            "history_coverage": cov,
            "interval_inflation": float(interval_inflation),
            "consistency_regularization": consistency_stats,
            "continuity_smoothing": continuity_stats,
            "robust_cross_member_blend": robust_blend_stats,
            "combo_selection": combo_meta,
            "combo_table": combo_table.to_dict(orient="records") if not combo_table.empty else [],
            "horizon_weighting": {
                "enabled": True,
                "recency_gap_years": float(recency_gap_years),
                "coverage_ratio": float(coverage_ratio),
                "profiles": horizon_weight_profiles,
            },
            "pseudo_climate_delta_fallback": {
                "enabled": len(pseudo_delta_models) > 0,
                "models": sorted(set(pseudo_delta_models)),
                "count": int(len(sorted(set(pseudo_delta_models)))),
            },
            "temp_min_warming_rate_c_per_year": float(min_temp_warming_rate_c_per_year) if variable == "temp" else 0.0,
            "temp_trend_adjustment_mean_c": float(trend_adjust_mean),
            "temp_trend_adjustment_max_c": float(trend_adjust_max),
        },
        score_df,
    )


def forecast_shape_objective(
    out_df: pd.DataFrame,
    obs_monthly: pd.Series,
    variable: str,
) -> dict[str, float]:
    if out_df is None or out_df.empty:
        return {
            "objective": float("inf"),
            "jump_q95": float("inf"),
            "jump_max": float("inf"),
            "ann_jump_q95": float("inf"),
            "ann_jump_max": float("inf"),
            "std_penalty": float("inf"),
            "amp_penalty": float("inf"),
        }

    fc = out_df[out_df["is_forecast"] == True][["ds", "yhat"]].copy()
    if fc.empty:
        return {
            "objective": float("inf"),
            "jump_q95": float("inf"),
            "jump_max": float("inf"),
            "ann_jump_q95": float("inf"),
            "ann_jump_max": float("inf"),
            "std_penalty": float("inf"),
            "amp_penalty": float("inf"),
        }

    fc["ds"] = pd.to_datetime(fc["ds"], errors="coerce")
    fc["yhat"] = pd.to_numeric(fc["yhat"], errors="coerce")
    fc = fc.dropna(subset=["ds", "yhat"]).sort_values("ds")
    y = fc["yhat"].values.astype(float)
    if len(y) < 3:
        return {
            "objective": float("inf"),
            "jump_q95": float("inf"),
            "jump_max": float("inf"),
            "ann_jump_q95": float("inf"),
            "ann_jump_max": float("inf"),
            "std_penalty": float("inf"),
            "amp_penalty": float("inf"),
        }

    d = np.abs(np.diff(y))
    jump_q95 = float(np.quantile(d, 0.95)) if len(d) else 0.0
    jump_max = float(np.max(d)) if len(d) else 0.0

    ann = fc.assign(year=fc["ds"].dt.year).groupby("year", as_index=False)["yhat"].mean()
    ad = np.abs(np.diff(ann["yhat"].values.astype(float)))
    ann_jump_q95 = float(np.quantile(ad, 0.95)) if len(ad) else 0.0
    ann_jump_max = float(np.max(ad)) if len(ad) else 0.0

    obs = pd.to_numeric(obs_monthly, errors="coerce").dropna().sort_index()
    obs_scale = float(np.nanmedian(np.abs(obs.values))) if len(obs) else 1.0
    obs_scale = float(max(obs_scale, 1.0))
    obs_std = float(np.nanstd(obs.values)) if len(obs) else float(np.nanstd(y))
    obs_std = float(max(obs_std, 1e-6))
    fc_std = float(np.nanstd(y))
    std_ratio = fc_std / obs_std
    std_pen = float(abs(np.log(max(std_ratio, 1e-6))))

    amp_pen = 0.0
    month_clim = smooth_month_climatology(obs, variable=variable)
    if len(month_clim) >= 6:
        obs_vals = np.array([float(v) for _, v in sorted(month_clim.items())], dtype=float)
        amp_obs = float(np.quantile(obs_vals, 0.90) - np.quantile(obs_vals, 0.10))
        fc_mm = fc.groupby(fc["ds"].dt.month)["yhat"].mean().dropna()
        if len(fc_mm) >= 6:
            fc_vals = fc_mm.values.astype(float)
            amp_fc = float(np.quantile(fc_vals, 0.90) - np.quantile(fc_vals, 0.10))
            amp_pen = float(abs(np.log((amp_fc + 1e-6) / max(amp_obs, 1e-6))))

    var = canonical_variable_name(variable)
    if var == "temp":
        objective = jump_q95 + 0.55 * ann_jump_q95 + 0.22 * obs_scale * std_pen + 0.20 * obs_scale * amp_pen
    elif var == "humidity":
        objective = jump_q95 + 0.55 * ann_jump_q95 + 0.18 * obs_scale * std_pen + 0.14 * obs_scale * amp_pen
    elif var == "precip":
        objective = jump_q95 + 0.45 * ann_jump_q95 + 0.08 * obs_scale * std_pen + 0.06 * obs_scale * amp_pen
    else:
        objective = jump_q95 + 0.50 * ann_jump_q95 + 0.15 * obs_scale * std_pen + 0.10 * obs_scale * amp_pen

    return {
        "objective": float(objective),
        "jump_q95": float(jump_q95),
        "jump_max": float(jump_max),
        "ann_jump_q95": float(ann_jump_q95),
        "ann_jump_max": float(ann_jump_max),
        "std_penalty": float(std_pen),
        "amp_penalty": float(amp_pen),
    }


def tune_force_combine_count(
    variable: str,
    obs_monthly: pd.Series,
    members: list[tuple[ModelScore, pd.DataFrame]],
    forecast_start_year: int,
    target_year: int,
    interval_z: float,
    min_w: float,
    max_w: float,
    auto_select_combination_flag: bool,
    max_combo_size: int,
    combo_complexity_penalty: float,
    min_temp_warming_rate_c_per_year: float,
    candidate_counts: list[int],
    fallback_force_count: int,
    obs_all: pd.DataFrame | None = None,
    humidity_forecast_df: pd.DataFrame | None = None,
    qc_ok_value: str = "ok",
    tune_continuity_alpha: bool = True,
    continuity_alpha_overrides: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    cands = [int(x) for x in candidate_counts if int(x) >= 2]
    if int(fallback_force_count) >= 2 and int(fallback_force_count) not in cands:
        cands.append(int(fallback_force_count))
    cands = sorted(set(cands))
    if not cands:
        cands = [max(0, int(fallback_force_count))]

    ref_end = pd.Timestamp(year=max(int(forecast_start_year) - 1, 1), month=12, day=1)
    cov = monthly_history_coverage(pd.to_numeric(obs_monthly, errors="coerce"), reference_end=ref_end)
    if bool(tune_continuity_alpha):
        if continuity_alpha_overrides:
            alpha_opts = sorted(set(float(np.clip(a, 0.0, 0.35)) for a in continuity_alpha_overrides))
        else:
            alpha_opts = continuity_alpha_candidates(variable=variable, coverage=cov)
        alpha_opts = [None] + [float(a) for a in alpha_opts if float(a) > 1e-9]
    else:
        alpha_opts = [None]

    rows: list[dict[str, Any]] = []
    candidates_eval: list[dict[str, Any]] = []

    for cnt in cands:
        for alpha_opt in alpha_opts:
            out_df, rep_extra, score_df = blend_variable(
                variable=variable,
                obs_monthly=obs_monthly,
                members=members,
                forecast_start_year=int(forecast_start_year),
                target_year=int(target_year),
                interval_z=float(interval_z),
                min_w=float(min_w),
                max_w=float(max_w),
                auto_select_combination_flag=bool(auto_select_combination_flag),
                max_combo_size=int(max_combo_size),
                combo_complexity_penalty=float(combo_complexity_penalty),
                force_combine_count=int(cnt),
                min_temp_warming_rate_c_per_year=float(min_temp_warming_rate_c_per_year),
                continuity_alpha_override=alpha_opt,
            )
            if out_df.empty:
                continue
            eval_df = out_df
            if (
                canonical_variable_name(variable) == "temp"
                and obs_all is not None
                and humidity_forecast_df is not None
                and not humidity_forecast_df.empty
            ):
                try:
                    eval_df, _ = coupled_temp_from_humidity_forecast(
                        obs=obs_all,
                        temp_df=out_df,
                        humidity_df=humidity_forecast_df,
                        ok_value=qc_ok_value,
                        forecast_start_year=int(forecast_start_year),
                        target_year=int(target_year),
                        interval_z=float(interval_z),
                    )
                except Exception:
                    eval_df = out_df

            # Keep tuning objective aligned with final pipeline:
            # after temp-humidity coupling, apply the same forecast-only continuity step.
            if canonical_variable_name(variable) == "temp" and eval_df is not None and not eval_df.empty:
                try:
                    fc_mask = eval_df["is_forecast"] == True
                    if bool(fc_mask.any()):
                        alpha_after: float | None = None
                        if alpha_opt is not None and float(alpha_opt) > 1e-9:
                            alpha_after = float(alpha_opt)
                        else:
                            cst_local = rep_extra.get("continuity_smoothing", {}) if isinstance(rep_extra, dict) else {}
                            if isinstance(cst_local, dict):
                                aa2 = float(cst_local.get("selected_alpha", 0.0))
                                if aa2 > 1e-9:
                                    alpha_after = aa2
                        fc_view = eval_df.loc[fc_mask, ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                        fc_sm, _ = apply_continuity_smoothing(
                            blend=fc_view,
                            variable=variable,
                            coverage=cov,
                            alpha_override=alpha_after,
                        )
                        eval_df = eval_df.copy()
                        eval_df.loc[fc_mask, "yhat"] = fc_sm["yhat"].values
                        eval_df.loc[fc_mask, "yhat_lower"] = fc_sm["yhat_lower"].values
                        eval_df.loc[fc_mask, "yhat_upper"] = fc_sm["yhat_upper"].values
                except Exception:
                    pass

            met = forecast_shape_objective(out_df=eval_df, obs_monthly=obs_monthly, variable=variable)
            j95m = float(met.get("jump_q95", float("inf")))
            jmxm = float(met.get("jump_max", float("inf")))
            a95m = float(met.get("ann_jump_q95", float("inf")))
            objm = float(met.get("objective", float("inf")))
            tune_obj = objm
            row = {
                "force_combine_count": int(cnt),
                "continuity_alpha_override": (float(alpha_opt) if alpha_opt is not None else np.nan),
                "tuning_objective": float(tune_obj),
            }
            row.update({k: float(v) for k, v in met.items()})
            rows.append(row)
            candidates_eval.append(
                {
                    "count": int(cnt),
                    "alpha_override": (float(alpha_opt) if alpha_opt is not None else None),
                    "out_df": out_df,
                    "rep_extra": rep_extra,
                    "score_df": score_df,
                    "met": met,
                    "tuning_objective": float(tune_obj),
                }
            )

    if not candidates_eval:
        out_df, rep_extra, score_df = blend_variable(
            variable=variable,
            obs_monthly=obs_monthly,
            members=members,
            forecast_start_year=int(forecast_start_year),
            target_year=int(target_year),
            interval_z=float(interval_z),
            min_w=float(min_w),
            max_w=float(max_w),
            auto_select_combination_flag=bool(auto_select_combination_flag),
            max_combo_size=int(max_combo_size),
            combo_complexity_penalty=float(combo_complexity_penalty),
            force_combine_count=int(fallback_force_count),
            min_temp_warming_rate_c_per_year=float(min_temp_warming_rate_c_per_year),
            continuity_alpha_override=None,
        )
        rep_extra = dict(rep_extra)
        rep_extra["force_combine_tuning"] = {
            "enabled": True,
            "reason": "all_candidates_failed",
            "candidate_counts": cands,
            "continuity_alpha_candidates": [float(a) for a in alpha_opts if a is not None],
            "selected_force_combine_count": int(fallback_force_count),
            "selected_continuity_alpha_override": None,
            "selected_objective": None,
            "baseline_force_combine_count": int(fallback_force_count),
            "baseline_continuity_alpha_override": None,
            "guardrails": {},
            "table": sorted(rows, key=lambda r: float(r.get("objective", float("inf")))),
        }
        return out_df, rep_extra, score_df

    baseline_count = int(fallback_force_count) if int(fallback_force_count) >= 2 else int(max(cands))
    base_eval = next(
        (r for r in candidates_eval if int(r["count"]) == baseline_count and r.get("alpha_override") is None),
        None,
    )
    if base_eval is None:
        base_eval = next((r for r in candidates_eval if int(r["count"]) == baseline_count), None)
    if base_eval is None:
        base_eval = max(
            candidates_eval,
            key=lambda r: (
                int(r.get("count", 0)),
                0 if r.get("alpha_override") is None else 1,
            ),
        )
        baseline_count = int(base_eval["count"])
    base_met = dict(base_eval.get("met", {}))
    base_jump_q95 = float(base_met.get("jump_q95", float("inf")))
    base_jump_max = float(base_met.get("jump_max", float("inf")))
    base_ann_q95 = float(base_met.get("ann_jump_q95", float("inf")))

    f_j95, f_jmx, f_a95 = 1.00, 1.05, 1.05
    jump_q95_limit = float(base_jump_q95 * f_j95 + 1e-9)
    jump_max_limit = float(base_jump_max * f_jmx + 1e-9)
    ann_q95_limit = float(base_ann_q95 * f_a95 + 1e-9)

    eligible: list[dict[str, Any]] = []
    for cand in candidates_eval:
        met = dict(cand.get("met", {}))
        j95 = float(met.get("jump_q95", float("inf")))
        jmx = float(met.get("jump_max", float("inf")))
        aq95 = float(met.get("ann_jump_q95", float("inf")))
        if j95 <= jump_q95_limit and jmx <= jump_max_limit and aq95 <= ann_q95_limit:
            eligible.append(cand)
    if not any(int(c["count"]) == baseline_count for c in eligible):
        eligible.append(base_eval)

    best_eval = min(
        eligible,
        key=lambda r: (
            float(r.get("tuning_objective", float("inf"))),
            float(dict(r.get("met", {})).get("jump_q95", float("inf"))),
            abs(float(r.get("alpha_override", 0.0))) if r.get("alpha_override") is not None else 0.0,
            -int(r.get("count", 0)),
        ),
    )
    out_best = best_eval["out_df"]
    rep_best = best_eval["rep_extra"]
    score_best = best_eval["score_df"]
    met_best = dict(best_eval.get("met", {}))
    cnt_best = int(best_eval.get("count", baseline_count))

    rep_best = dict(rep_best)
    rep_best["force_combine_tuning"] = {
        "enabled": True,
        "reason": "ok",
        "candidate_counts": cands,
        "continuity_alpha_candidates": [float(a) for a in alpha_opts if a is not None],
        "selected_force_combine_count": int(cnt_best),
        "selected_continuity_alpha_override": best_eval.get("alpha_override"),
        "selected_objective": float(met_best.get("objective", float("nan"))),
        "selected_tuning_objective": float(best_eval.get("tuning_objective", float("nan"))),
        "baseline_force_combine_count": int(baseline_count),
        "baseline_continuity_alpha_override": base_eval.get("alpha_override"),
        "baseline_tuning_objective": float(base_eval.get("tuning_objective", float("nan"))),
        "guardrails": {
            "jump_q95_max": float(jump_q95_limit),
            "jump_max_max": float(jump_max_limit),
            "ann_jump_q95_max": float(ann_q95_limit),
        },
        "table": sorted(rows, key=lambda r: float(r.get("tuning_objective", float("inf")))),
    }
    return out_best, rep_best, score_best


def plot_forecast(df: pd.DataFrame, variable: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    h = df[df["is_forecast"] == False]
    f = df[df["is_forecast"] == True]
    h = h.sort_values("ds").reset_index(drop=True)
    f = f.sort_values("ds").reset_index(drop=True)
    x_h = np.arange(len(h), dtype=float)
    x_f = np.arange(len(h), len(h) + len(f), dtype=float)

    def xticks_for_dense_axis(hist_df: pd.DataFrame, fc_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        pts: list[tuple[float, str]] = []
        if not hist_df.empty:
            hh = hist_df.copy()
            hh["year"] = pd.to_datetime(hh["ds"]).dt.year
            first_per_year = hh.groupby("year", as_index=False).first()
            for _, r in first_per_year.iterrows():
                idx = int(hist_df.index[hist_df["ds"] == r["ds"]][0])
                pts.append((float(idx), str(int(r["year"]))))
        if not fc_df.empty:
            ff = fc_df.copy()
            ff["year"] = pd.to_datetime(ff["ds"]).dt.year
            first_per_year = ff.groupby("year", as_index=False).first()
            for _, r in first_per_year.iterrows():
                idx = int(fc_df.index[fc_df["ds"] == r["ds"]][0]) + len(hist_df)
                pts.append((float(idx), str(int(r["year"]))))
        if not pts:
            return np.array([], dtype=float), []
        # Keep labels readable.
        if len(pts) > 12:
            step = int(np.ceil(len(pts) / 12.0))
            pts = [p for i, p in enumerate(pts) if i % step == 0 or i == len(pts) - 1]
        ticks = np.array([p[0] for p in pts], dtype=float)
        labels = [p[1] for p in pts]
        return ticks, labels

    if not h.empty:
        ax.plot(x_h, h["actual"], color="#1f77b4", linewidth=1.2, label="observed")
    if not f.empty:
        ax.plot(x_f, f["yhat"], color="#d62728", linewidth=2.0, label="best meta forecast")
        ax.fill_between(
            x_f,
            f["yhat_lower"],
            f["yhat_upper"],
            color="#d62728",
            alpha=0.15,
            label="interval",
        )
    if not h.empty and not f.empty:
        # Visual bridge between history and forecast to avoid harsh discontinuity in sparse histories.
        h_last = h.iloc[-1]
        f_first = f.iloc[0]
        ax.plot(
            [x_h[-1], x_f[0]],
            [h_last["actual"], f_first["yhat"]],
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
            label="history-forecast bridge",
        )
    if not h.empty and not f.empty:
        ax.axvline(float(len(h) - 0.5), color="#555555", linestyle="--", linewidth=1.0)
    ticks, labels = xticks_for_dense_axis(h, f)
    if len(ticks):
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=0)
    ax.set_title(f"Best Meta Ensemble Forecast - {variable} (monthly)")
    ax.set_xlabel("year (continuous axis)")
    ax.set_ylabel(f"value ({infer_unit(variable)})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def write_annual_compare(out: Path, temp_fc: pd.DataFrame | None, hum_fc: pd.DataFrame | None, start_year: int, target_year: int) -> None:
    if temp_fc is None or hum_fc is None:
        return
    t = temp_fc[(temp_fc["is_forecast"] == True)].copy()
    h = hum_fc[(hum_fc["is_forecast"] == True)].copy()
    t["year"] = pd.to_datetime(t["ds"]).dt.year
    h["year"] = pd.to_datetime(h["ds"]).dt.year
    t = t[(t["year"] >= start_year) & (t["year"] <= target_year)]
    h = h[(h["year"] >= start_year) & (h["year"] <= target_year)]
    if t.empty or h.empty:
        return

    at = t.groupby("year", as_index=False).agg(
        temp_yhat_mean=("yhat", "mean"),
        temp_lower_mean=("yhat_lower", "mean"),
        temp_upper_mean=("yhat_upper", "mean"),
        temp_climate_delta_mean=("climate_delta", "mean"),
    )
    ah = h.groupby("year", as_index=False).agg(
        humidity_yhat_mean=("yhat", "mean"),
        humidity_lower_mean=("yhat_lower", "mean"),
        humidity_upper_mean=("yhat_upper", "mean"),
        humidity_climate_delta_mean=("climate_delta", "mean"),
    )
    ann = at.merge(ah, on="year", how="inner").sort_values("year")
    corr_n = int(len(ann))
    corr = float(ann["temp_yhat_mean"].corr(ann["humidity_yhat_mean"])) if corr_n >= 3 else np.nan
    ann["temp_humidity_corr_same_year"] = corr
    ann["temp_humidity_corr_n_years"] = corr_n

    csv_path = out / f"annual_compare_temp_humidity_{start_year}_{target_year}_best.csv"
    md_path = out / f"annual_compare_temp_humidity_{start_year}_{target_year}_best.md"
    png_t = out / f"annual_compare_temp_{start_year}_{target_year}_best.png"
    png_h = out / f"annual_compare_humidity_{start_year}_{target_year}_best.png"

    ann.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Yillik Sicaklik-Nem Karsilastirmasi (Best Meta)\n\n" + ann.round(4).to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(10, 4.8))
    plt.plot(ann["year"], ann["temp_yhat_mean"], marker="o", color="#d62728")
    plt.title("Yillik Ortalama Sicaklik Tahmini (Best Meta)")
    plt.xlabel("Yil")
    plt.ylabel("Sicaklik (C)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(png_t, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    plt.plot(ann["year"], ann["humidity_yhat_mean"], marker="o", color="#1f77b4")
    plt.title("Yillik Ortalama Nem Tahmini (Best Meta)")
    plt.xlabel("Yil")
    plt.ylabel("Nem (%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(png_h, dpi=160)
    plt.close()


def write_annual_compare_all_years(out: Path, temp_df: pd.DataFrame | None, hum_df: pd.DataFrame | None) -> None:
    if temp_df is None or hum_df is None:
        return

    t = temp_df.copy()
    h = hum_df.copy()
    t["year"] = pd.to_datetime(t["ds"]).dt.year
    h["year"] = pd.to_datetime(h["ds"]).dt.year

    t_ann = t.groupby(["year", "is_forecast"], as_index=False).agg(
        temp_yhat_mean=("yhat", "mean"),
        temp_actual_mean=("actual", "mean"),
        temp_lower_mean=("yhat_lower", "mean"),
        temp_upper_mean=("yhat_upper", "mean"),
        temp_points=("yhat", "count"),
        temp_actual_points=("actual", "count"),
    )
    h_ann = h.groupby(["year", "is_forecast"], as_index=False).agg(
        humidity_yhat_mean=("yhat", "mean"),
        humidity_actual_mean=("actual", "mean"),
        humidity_lower_mean=("yhat_lower", "mean"),
        humidity_upper_mean=("yhat_upper", "mean"),
        humidity_points=("yhat", "count"),
        humidity_actual_points=("actual", "count"),
    )
    ann = t_ann.merge(h_ann, on=["year", "is_forecast"], how="outer").sort_values(["year", "is_forecast"])
    ann["phase"] = np.where(ann["is_forecast"].fillna(False), "forecast", "history")

    # Year-level correlation on model means where both exist.
    corr_src = (
        ann.groupby("year", as_index=False)
        .agg(temp_yhat_mean=("temp_yhat_mean", "mean"), humidity_yhat_mean=("humidity_yhat_mean", "mean"))
        .dropna()
    )
    corr_n = int(len(corr_src))
    corr = float(corr_src["temp_yhat_mean"].corr(corr_src["humidity_yhat_mean"])) if corr_n >= 3 else np.nan
    ann["temp_humidity_corr_all_years"] = corr
    ann["temp_humidity_corr_n_years"] = corr_n

    hist_src = (
        ann[ann["phase"] == "history"]
        .groupby("year", as_index=False)
        .agg(temp_actual_mean=("temp_actual_mean", "mean"), humidity_actual_mean=("humidity_actual_mean", "mean"))
        .dropna()
    )
    hist_corr_n = int(len(hist_src))
    hist_corr = float(hist_src["temp_actual_mean"].corr(hist_src["humidity_actual_mean"])) if hist_corr_n >= 3 else np.nan
    ann["temp_humidity_corr_history_obs"] = hist_corr
    ann["temp_humidity_corr_history_n_years"] = hist_corr_n

    fc_src = (
        ann[ann["phase"] == "forecast"]
        .groupby("year", as_index=False)
        .agg(temp_yhat_mean=("temp_yhat_mean", "mean"), humidity_yhat_mean=("humidity_yhat_mean", "mean"))
        .dropna()
    )
    fc_corr_n = int(len(fc_src))
    fc_corr = float(fc_src["temp_yhat_mean"].corr(fc_src["humidity_yhat_mean"])) if fc_corr_n >= 3 else np.nan
    ann["temp_humidity_corr_forecast"] = fc_corr
    ann["temp_humidity_corr_forecast_n_years"] = fc_corr_n

    csv_path = out / "annual_compare_temp_humidity_all_years_best.csv"
    md_path = out / "annual_compare_temp_humidity_all_years_best.md"
    png_t = out / "annual_compare_temp_all_years_best.png"
    png_h = out / "annual_compare_humidity_all_years_best.png"

    ann.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Tum Yillar Sicaklik-Nem Karsilastirmasi (Best Meta)\n\n" + ann.round(4).to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    def plot_yearly_continuous(
        years: pd.Series,
        values: pd.Series,
        year_to_pos: dict[int, int],
        color: str,
        label: str,
        line_width: float = 1.6,
    ) -> None:
        d = pd.DataFrame({"year": years, "val": values}).dropna().sort_values("year")
        if d.empty:
            return
        xs = np.array([float(year_to_pos[int(y)]) for y in d["year"].astype(int).tolist()], dtype=float)
        ys = d["val"].astype(float).values
        plt.plot(xs, ys, color=color, linewidth=line_width, alpha=0.92, label=label)
        plt.scatter(xs, ys, color=color, s=16, alpha=0.96)

    def ticks_from_years(ordered_years: list[int], max_labels: int = 12) -> tuple[np.ndarray, list[str]]:
        if not ordered_years:
            return np.array([], dtype=float), []
        pts = list(enumerate(ordered_years))
        if len(pts) > max_labels:
            step = int(np.ceil(len(pts) / float(max_labels)))
            pts = [p for i, p in enumerate(pts) if i % step == 0 or i == len(pts) - 1]
        ticks = np.array([float(i) for i, _ in pts], dtype=float)
        labels = [str(int(y)) for _, y in pts]
        return ticks, labels

    # Temperature all-years chart.
    t_hist = ann[(ann["phase"] == "history") & ann["temp_actual_mean"].notna()].copy()
    t_fc = ann[(ann["phase"] == "forecast") & ann["temp_yhat_mean"].notna()].copy()
    t_years = sorted(set(t_hist["year"].astype(int).tolist()) | set(t_fc["year"].astype(int).tolist()))
    t_pos = {int(y): i for i, y in enumerate(t_years)}
    plt.figure(figsize=(11, 4.8))
    if not t_hist.empty:
        plot_yearly_continuous(
            t_hist["year"],
            t_hist["temp_actual_mean"],
            year_to_pos=t_pos,
            color="#1f77b4",
            label="Sicaklik gozlem (yillik)",
            line_width=1.5,
        )
    if not t_fc.empty:
        plot_yearly_continuous(
            t_fc["year"],
            t_fc["temp_yhat_mean"],
            year_to_pos=t_pos,
            color="#d62728",
            label="Sicaklik tahmin (yillik)",
            line_width=2.0,
        )
    tticks, tlabels = ticks_from_years(t_years)
    if len(tticks):
        plt.xticks(tticks, tlabels)
    plt.title("Tum Yillar Sicaklik (Best Meta)")
    plt.xlabel("Yil (continuous axis)")
    plt.ylabel("Sicaklik (C)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(png_t, dpi=160)
    plt.close()

    # Humidity all-years chart.
    h_hist = ann[(ann["phase"] == "history") & ann["humidity_actual_mean"].notna()].copy()
    h_fc = ann[(ann["phase"] == "forecast") & ann["humidity_yhat_mean"].notna()].copy()
    h_years = sorted(set(h_hist["year"].astype(int).tolist()) | set(h_fc["year"].astype(int).tolist()))
    h_pos = {int(y): i for i, y in enumerate(h_years)}
    plt.figure(figsize=(11, 4.8))
    if not h_hist.empty:
        plot_yearly_continuous(
            h_hist["year"],
            h_hist["humidity_actual_mean"],
            year_to_pos=h_pos,
            color="#1f77b4",
            label="Nem gozlem (yillik)",
            line_width=1.5,
        )
    if not h_fc.empty:
        plot_yearly_continuous(
            h_fc["year"],
            h_fc["humidity_yhat_mean"],
            year_to_pos=h_pos,
            color="#d62728",
            label="Nem tahmin (yillik)",
            line_width=2.0,
        )
    hticks, hlabels = ticks_from_years(h_years)
    if len(hticks):
        plt.xticks(hticks, hlabels)
    plt.title("Tum Yillar Nem (Best Meta)")
    plt.xlabel("Yil (continuous axis)")
    plt.ylabel("Nem (%)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(png_h, dpi=160)
    plt.close()


def write_annual_pairwise_correlations(
    out: Path,
    out_map: dict[str, pd.DataFrame],
    forecast_start_year: int,
    target_year: int,
) -> None:
    vars_use = sorted([str(v) for v in out_map.keys() if out_map.get(v) is not None and not out_map.get(v).empty])
    if len(vars_use) < 2:
        return

    series_map: dict[str, pd.DataFrame] = {}
    for var in vars_use:
        d = out_map[var].copy()
        d["year"] = pd.to_datetime(d["ds"], errors="coerce").dt.year
        d = d.dropna(subset=["year"])
        g = d.groupby(["year", "is_forecast"], as_index=False).agg(
            yhat_mean=("yhat", "mean"),
            actual_mean=("actual", "mean"),
        )
        series_map[var] = g

    rows: list[dict[str, Any]] = []

    def pair_corr(s1: pd.DataFrame, s2: pd.DataFrame, col1: str, col2: str) -> tuple[float, int]:
        a = s1.rename(columns={col1: "v1"})
        b = s2.rename(columns={col2: "v2"})
        m = a.merge(b, on="year", how="inner")
        m = m.dropna(subset=["v1", "v2"])
        n = int(len(m))
        if n < 3:
            return float("nan"), n
        return float(m["v1"].corr(m["v2"])), n

    scopes = [
        ("all_years", None, None),
        ("forecast_window", int(forecast_start_year), int(target_year)),
    ]
    phases = [
        ("forecast", True, "yhat_mean", "yhat_mean"),
        ("history", False, "actual_mean", "actual_mean"),
    ]

    for scope_name, y0, y1 in scopes:
        for phase_name, fc_flag, c1, c2 in phases:
            for a, b in itertools.combinations(vars_use, 2):
                sa = series_map[a]
                sb = series_map[b]
                sa = sa[sa["is_forecast"] == fc_flag].copy()
                sb = sb[sb["is_forecast"] == fc_flag].copy()
                if y0 is not None and y1 is not None:
                    sa = sa[(sa["year"] >= y0) & (sa["year"] <= y1)]
                    sb = sb[(sb["year"] >= y0) & (sb["year"] <= y1)]
                corr, n = pair_corr(sa[["year", c1]], sb[["year", c2]], c1, c2)
                rows.append(
                    {
                        "scope": scope_name,
                        "phase": phase_name,
                        "variable_a": a,
                        "variable_b": b,
                        "corr": corr,
                        "n_years": int(n),
                    }
                )

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(["scope", "phase", "variable_a", "variable_b"]).reset_index(drop=True)
    csv_path = out / "annual_pairwise_correlations_best.csv"
    md_path = out / "annual_pairwise_correlations_best.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Yillik Pairwise Korelasyonlar (Best Meta)\n\n" + df.round(4).to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    forecast_start_year = int(args.forecast_start_year) if args.forecast_start_year is not None else int(args.walkforward_start_year)

    raw = read_table(args.observations)
    obs, kind = normalize_observations(raw, args)
    if to_bool(args.enable_temp_proxy):
        obs, proxy_stats = augment_temp_from_humidity_proxy(obs, ok_value=args.qc_ok_value)
    else:
        proxy_stats = {
            "enabled": False,
            "reason": "disabled_by_arg",
            "added_points": 0,
            "temp_months_before": int(monthly_history_coverage(monthly_from_obs_no_fill(obs, "temp", args.qc_ok_value))["months_with_data"])
            if "temp" in set(obs["variable"].astype(str).unique().tolist())
            else 0,
            "temp_months_after": int(monthly_history_coverage(monthly_from_obs_no_fill(obs, "temp", args.qc_ok_value))["months_with_data"])
            if "temp" in set(obs["variable"].astype(str).unique().tolist())
            else 0,
        }
    vars_use = requested_variables(obs, args.variables)
    if not vars_use:
        raise SystemExit("No variables selected.")
    auto_tune_force = to_bool(args.auto_tune_force_combine)
    auto_tune_continuity_alpha = to_bool(args.auto_tune_continuity_alpha)
    force_candidates = [int(x) for x in parse_int_csv(args.force_combine_candidates) if int(x) >= 2]
    continuity_alpha_override_candidates = [float(x) for x in parse_float_csv(args.continuity_alpha_candidates)]
    if int(args.force_combine_count) >= 2 and int(args.force_combine_count) not in force_candidates:
        force_candidates.append(int(args.force_combine_count))
    force_candidates = sorted(set(force_candidates))
    if not force_candidates:
        force_candidates = [8, 10, 12, 15]

    out = args.output_dir
    base_dir = out / args.base_dir_name
    fc_dir = out / "forecasts"
    rep_dir = out / "reports"
    ch_dir = out / "charts"
    lb_dir = out / "leaderboards"
    for d in [out, base_dir, fc_dir, rep_dir, ch_dir, lb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (out / "proxy_temp_augmentation.json").write_text(json.dumps(proxy_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    maybe_run_base_models(args, base_dir=base_dir)
    scores = load_model_scores(base_dir=base_dir, target_year=args.target_year, start_year=args.walkforward_start_year)
    if not scores:
        raise SystemExit("No base model scores were found. Check base model outputs.")

    climate_cfg = climate_from_args(args)

    index_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    out_map: dict[str, pd.DataFrame] = {}

    for variable in vars_use:
        obs_m = choose_monthly_observed_series(obs, variable=variable, ok_value=args.qc_ok_value)
        ref_end = pd.Timestamp(year=max(int(forecast_start_year) - 1, 1), month=12, day=1)
        cov = monthly_history_coverage(obs_m, reference_end=ref_end)
        coverage_rows.append(
            {
                "variable": variable,
                "months_with_data": int(cov["months_with_data"]),
                "years_with_data": int(cov["years_with_data"]),
                "span_months": int(cov["span_months"]),
                "coverage_ratio": float(cov["coverage_ratio"]),
                "month_of_year_coverage": int(cov["month_of_year_coverage"]),
                "max_internal_gap_months": int(cov["max_internal_gap_months"]),
                "history_start": cov["history_start"],
                "history_end": cov["history_end"],
                "quality_flag": cov["quality_flag"],
                "status": "candidate",
            }
        )
        if obs_m.empty:
            coverage_rows[-1]["status"] = "skipped_no_history"
            continue

        ms_list = [m for m in scores if m.variable == variable and str(m.frequency).upper().startswith("M")]
        if not ms_list:
            coverage_rows[-1]["status"] = "skipped_no_monthly_models"
            continue

        members: list[tuple[ModelScore, pd.DataFrame]] = []
        for ms in ms_list:
            p = resolve_path(ms.forecast_csv, Path.cwd())
            if not p.exists():
                continue
            try:
                dfm = standardize_forecast_frame(p, variable=variable)
                dfm = apply_climate_to_member(dfm, variable=variable, climate_cfg_global=climate_cfg)
            except Exception:
                continue
            members.append((ms, dfm))

        if not members:
            coverage_rows[-1]["status"] = "skipped_no_member_forecast"
            continue

        climo_stats = {"enabled": False, "reason": "not_built"}
        climo_df, climo_stats = build_climo_decay_member(
            variable=variable,
            obs_monthly=obs_m,
            forecast_start_year=int(forecast_start_year),
            target_year=int(args.target_year),
            interval_z=float(args.interval_z),
            coverage=cov,
        )
        if not climo_df.empty:
            hh = climo_df[climo_df["is_forecast"] == False].dropna(subset=["actual", "yhat"]).copy()
            if not hh.empty:
                err = hh["actual"].values.astype(float) - hh["yhat"].values.astype(float)
                c_rmse = float(np.sqrt(np.mean(err**2)))
                c_bias = float(np.mean(err))
                c_std = float(np.std(err))
                c_ms = ModelScore(
                    model_key="climo_decay",
                    variable=variable,
                    frequency="MS",
                    forecast_csv="generated://climo_decay",
                    rmse=c_rmse,
                    bias=c_bias,
                    rmse_std=c_std,
                    has_cv=False,
                    source="generated:climo_decay",
                    raw={"model_strategy": "climo_decay_baseline", "stats": climo_stats},
                )
                members.append((c_ms, climo_df))

        super_ensemble_stats = {"enabled": False, "reason": "not_built", "base_member_count": len(members), "virtual_member_count": 0}
        virtual_members, super_ensemble_stats = build_virtual_member_variants(
            variable=variable,
            obs_monthly=obs_m,
            base_members=members,
        )
        if virtual_members:
            members.extend(virtual_members)

        if auto_tune_force:
            out_df, rep_extra, score_df = tune_force_combine_count(
                variable=variable,
                obs_monthly=obs_m,
                members=members,
                forecast_start_year=int(forecast_start_year),
                target_year=int(args.target_year),
                interval_z=float(args.interval_z),
                min_w=float(args.min_model_weight),
                max_w=float(args.max_model_weight),
                auto_select_combination_flag=to_bool(args.auto_select_combination),
                max_combo_size=int(max(1, args.max_combo_size)),
                combo_complexity_penalty=float(max(0.0, args.combo_complexity_penalty)),
                min_temp_warming_rate_c_per_year=(float(max(0.0, args.min_temp_warming_rate_c_per_year)) if variable == "temp" else 0.0),
                candidate_counts=force_candidates,
                fallback_force_count=int(max(0, args.force_combine_count)),
                obs_all=obs,
                humidity_forecast_df=out_map.get("humidity"),
                qc_ok_value=args.qc_ok_value,
                tune_continuity_alpha=bool(auto_tune_continuity_alpha),
                continuity_alpha_overrides=continuity_alpha_override_candidates,
            )
        else:
            out_df, rep_extra, score_df = blend_variable(
                variable=variable,
                obs_monthly=obs_m,
                members=members,
                forecast_start_year=int(forecast_start_year),
                target_year=int(args.target_year),
                interval_z=float(args.interval_z),
                min_w=float(args.min_model_weight),
                max_w=float(args.max_model_weight),
                auto_select_combination_flag=to_bool(args.auto_select_combination),
                max_combo_size=int(max(1, args.max_combo_size)),
                combo_complexity_penalty=float(max(0.0, args.combo_complexity_penalty)),
                force_combine_count=int(max(0, args.force_combine_count)),
                min_temp_warming_rate_c_per_year=(float(max(0.0, args.min_temp_warming_rate_c_per_year)) if variable == "temp" else 0.0),
                continuity_alpha_override=None,
            )
        if out_df.empty:
            coverage_rows[-1]["status"] = "skipped_empty_blend"
            continue
        coverage_rows[-1]["status"] = "fitted"

        coupled_stats = {"enabled": False, "reason": "not_applicable"}
        post_coupling_continuity = {"enabled": False, "reason": "not_applicable"}
        if variable == "temp" and "humidity" in out_map:
            out_df, coupled_stats = coupled_temp_from_humidity_forecast(
                obs=obs,
                temp_df=out_df,
                humidity_df=out_map["humidity"],
                ok_value=args.qc_ok_value,
                forecast_start_year=int(forecast_start_year),
                target_year=int(args.target_year),
                interval_z=float(args.interval_z),
            )
            fc_mask = out_df["is_forecast"] == True
            if bool(fc_mask.any()):
                alpha_opts: list[float | None] = [None]
                fct = rep_extra.get("force_combine_tuning", {}) if isinstance(rep_extra, dict) else {}
                if isinstance(fct, dict) and fct.get("selected_continuity_alpha_override") is not None:
                    aa = float(fct.get("selected_continuity_alpha_override"))
                    if aa > 1e-9:
                        alpha_opts.append(float(np.clip(aa, 0.0, 0.35)))
                cst = rep_extra.get("continuity_smoothing", {}) if isinstance(rep_extra, dict) else {}
                if isinstance(cst, dict):
                    aa2 = float(cst.get("selected_alpha", 0.0))
                    if aa2 > 1e-9:
                        alpha_opts.append(float(np.clip(aa2, 0.0, 0.35)))
                if canonical_variable_name(variable) == "temp":
                    # Broaden post-coupling search for temperature to reduce annual jaggedness
                    # while keeping monthly jump guardrails active.
                    alpha_opts.extend([0.22, 0.28, 0.32, 0.35])
                alpha_opts = list(dict.fromkeys(alpha_opts))

                base_fc = out_df.loc[fc_mask, ["ds", "yhat"]].copy()
                base_cm = continuity_metrics(base_fc["ds"], pd.to_numeric(base_fc["yhat"], errors="coerce").values.astype(float))
                base_j95 = float(base_cm.get("jump_q95", float("inf")))
                base_jmx = float(base_cm.get("jump_max", float("inf")))
                base_a95 = float(base_cm.get("ann_jump_q95", float("inf")))
                j95_lim = float(base_j95 * 1.02 + 1e-9)
                jmx_lim = float(base_jmx * 1.05 + 1e-9)
                a95_lim = float(base_a95 * 1.05 + 1e-9)

                if canonical_variable_name(variable) == "temp":
                    w_ann, w_max = 1.40, 0.03
                elif canonical_variable_name(variable) == "humidity":
                    w_ann, w_max = 1.20, 0.04
                elif canonical_variable_name(variable) == "precip":
                    w_ann, w_max = 0.80, 0.01
                else:
                    w_ann, w_max = 1.00, 0.02

                def post_couple_obj(cm: dict[str, float]) -> float:
                    return (
                        float(cm.get("jump_q95", float("inf")))
                        + float(w_ann) * float(cm.get("ann_jump_q95", float("inf")))
                        + float(w_max) * float(cm.get("jump_max", float("inf")))
                    )

                base_obj = float(post_couple_obj(base_cm))

                tuning_rows: list[dict[str, Any]] = []
                best_obj = float(base_obj)
                best_df = out_df
                best_stats = {"enabled": False, "reason": "baseline_after_coupling"}
                best_alpha = None

                for aopt in alpha_opts:
                    fc_view = out_df.loc[fc_mask, ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                    fc_sm, sm_stats = apply_continuity_smoothing(
                        blend=fc_view,
                        variable=variable,
                        coverage=cov,
                        alpha_override=aopt,
                    )
                    cand_df = out_df.copy()
                    cand_df.loc[fc_mask, "yhat"] = fc_sm["yhat"].values
                    cand_df.loc[fc_mask, "yhat_lower"] = fc_sm["yhat_lower"].values
                    cand_df.loc[fc_mask, "yhat_upper"] = fc_sm["yhat_upper"].values
                    cand_fc = cand_df.loc[fc_mask, ["ds", "yhat"]].copy()
                    met = continuity_metrics(
                        cand_fc["ds"],
                        pd.to_numeric(cand_fc["yhat"], errors="coerce").values.astype(float),
                    )
                    j95 = float(met.get("jump_q95", float("inf")))
                    jmx = float(met.get("jump_max", float("inf")))
                    a95 = float(met.get("ann_jump_q95", float("inf")))
                    obj = float(post_couple_obj(met))
                    eligible = bool(j95 <= j95_lim and jmx <= jmx_lim and a95 <= a95_lim)
                    tuning_rows.append(
                        {
                            "alpha_override": (float(aopt) if aopt is not None else np.nan),
                            "objective": float(obj),
                            "eligible": bool(eligible),
                            "jump_q95": float(j95),
                            "jump_max": float(jmx),
                            "ann_jump_q95": float(a95),
                            "enabled": bool(sm_stats.get("enabled", False)),
                            "selected_alpha": float(sm_stats.get("selected_alpha", 0.0)),
                            "reason": str(sm_stats.get("reason", "")),
                        }
                    )
                    if eligible and obj < best_obj - 1e-9:
                        best_obj = obj
                        best_df = cand_df
                        best_stats = sm_stats
                        best_alpha = aopt

                out_df = best_df
                post_coupling_continuity = dict(best_stats)
                post_coupling_continuity["tuning"] = {
                    "selected_alpha_override": (float(best_alpha) if best_alpha is not None else None),
                    "guardrails": {
                        "jump_q95_max": float(j95_lim),
                        "jump_max_max": float(jmx_lim),
                        "ann_jump_q95_max": float(a95_lim),
                    },
                    "table": sorted(tuning_rows, key=lambda r: float(r.get("objective", float("inf")))),
                }
        elif variable != "temp":
            fc_mask = out_df["is_forecast"] == True
            if bool(fc_mask.any()):
                fc_view = out_df.loc[fc_mask, ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                ds_fc = pd.to_datetime(fc_view["ds"], errors="coerce")
                y0 = pd.to_numeric(fc_view["yhat"], errors="coerce").values.astype(float)
                lo0 = pd.to_numeric(fc_view["yhat_lower"], errors="coerce").values.astype(float)
                hi0 = pd.to_numeric(fc_view["yhat_upper"], errors="coerce").values.astype(float)
                if not np.isfinite(y0).all():
                    post_coupling_continuity = {"enabled": False, "reason": "nonfinite_forecast"}
                else:
                    cst_local = rep_extra.get("continuity_smoothing", {}) if isinstance(rep_extra, dict) else {}
                    alpha_seed = 0.0
                    if isinstance(cst_local, dict):
                        alpha_seed = float(cst_local.get("selected_alpha", 0.0))

                    varc = canonical_variable_name(variable)
                    if varc == "humidity":
                        alpha_opts = [0.02, 0.04, 0.08, 0.12, 0.16, 0.18, 0.22]
                        w_ann, w_max = 0.75, 0.02
                        ann_lim_mul = 1.04
                        ann_rise_penalty = 2.40
                    elif varc == "precip":
                        alpha_opts = [0.03, 0.06, 0.09, 0.12, 0.14, 0.18]
                        w_ann, w_max = 0.65, 0.01
                        ann_lim_mul = 1.06
                        ann_rise_penalty = 0.80
                    else:
                        alpha_opts = [0.04, 0.08, 0.12, 0.16]
                        w_ann, w_max = 0.45, 0.02
                        ann_lim_mul = 1.07
                        ann_rise_penalty = 1.00
                    if alpha_seed > 1e-9:
                        alpha_opts.append(float(np.clip(alpha_seed, 0.0, 0.35)))
                    alpha_opts = sorted(set(float(np.clip(a, 0.0, 0.35)) for a in alpha_opts if float(a) > 1e-9))

                    base_cm = continuity_metrics(ds_fc, y0)
                    base_j95 = float(base_cm.get("jump_q95", float("inf")))
                    base_jmx = float(base_cm.get("jump_max", float("inf")))
                    base_a95 = float(base_cm.get("ann_jump_q95", float("inf")))
                    base_amp = float(max(base_cm.get("amp_p90_p10", 0.0), 1e-6))
                    base_mean = float(base_cm.get("mean", 0.0))
                    j95_lim = float(base_j95 * 1.01 + 1e-9)
                    jmx_lim = float(base_jmx * 1.03 + 1e-9)
                    a95_lim = float(base_a95 * ann_lim_mul + 1e-9)
                    amp_floor = float(0.72 * base_amp)
                    mean_tol = max(1.0, 0.02 * abs(base_mean))

                    def non_temp_obj(cm: dict[str, float]) -> float:
                        ann_val = float(cm.get("ann_jump_q95", float("inf")))
                        ann_rise = float(max(0.0, ann_val - base_a95))
                        return (
                            float(cm.get("jump_q95", float("inf")))
                            + float(w_ann) * ann_val
                            + float(w_max) * float(cm.get("jump_max", float("inf")))
                            + float(ann_rise_penalty) * ann_rise
                        )

                    base_obj = float(non_temp_obj(base_cm))
                    best_obj = float(base_obj)
                    best_alpha = 0.0
                    best_passes = 0
                    best_y = y0.copy()
                    best_cm = dict(base_cm)
                    tuning_rows: list[dict[str, Any]] = [
                        {
                            "alpha_override": np.nan,
                            "objective": float(base_obj),
                            "eligible": True,
                            "jump_q95": float(base_j95),
                            "jump_max": float(base_jmx),
                            "ann_jump_q95": float(base_a95),
                            "amp_p90_p10": float(base_amp),
                            "mean": float(base_mean),
                            "passes": 0,
                            "enabled": False,
                            "reason": "baseline",
                        }
                    ]

                    for aopt in alpha_opts:
                        passes = 3 if aopt >= 0.16 else 2
                        ys = median3_smooth(y0, alpha=float(aopt), passes=passes)
                        cm = continuity_metrics(ds_fc, ys)
                        j95 = float(cm.get("jump_q95", float("inf")))
                        jmx = float(cm.get("jump_max", float("inf")))
                        a95 = float(cm.get("ann_jump_q95", float("inf")))
                        amp = float(cm.get("amp_p90_p10", 0.0))
                        mm = float(cm.get("mean", 0.0))
                        obj = float(non_temp_obj(cm))
                        eligible = bool(
                            j95 <= j95_lim
                            and jmx <= jmx_lim
                            and a95 <= a95_lim
                            and amp >= amp_floor
                            and abs(mm - base_mean) <= mean_tol
                        )
                        tuning_rows.append(
                            {
                                "alpha_override": float(aopt),
                                "objective": float(obj),
                                "eligible": bool(eligible),
                                "jump_q95": float(j95),
                                "jump_max": float(jmx),
                                "ann_jump_q95": float(a95),
                                "amp_p90_p10": float(amp),
                                "mean": float(mm),
                                "passes": int(passes),
                                "enabled": bool(eligible and aopt > 1e-9),
                                "reason": ("ok" if eligible else "guardrail_reject"),
                            }
                        )
                        if eligible and obj < best_obj - 1e-9:
                            best_obj = float(obj)
                            best_alpha = float(aopt)
                            best_passes = int(passes)
                            best_y = ys
                            best_cm = dict(cm)

                    if best_alpha > 1e-9:
                        shift = best_y - y0
                        out_df.loc[fc_mask, "yhat"] = best_y
                        out_df.loc[fc_mask, "yhat_lower"] = lo0 + shift
                        out_df.loc[fc_mask, "yhat_upper"] = hi0 + shift
                        post_coupling_continuity = {
                            "enabled": True,
                            "reason": "ok",
                            "alpha_override": float(best_alpha),
                            "selected_alpha": float(best_alpha),
                            "passes": int(best_passes),
                            "before": {k: float(v) for k, v in base_cm.items()},
                            "after": {k: float(v) for k, v in best_cm.items()},
                            "mean_abs_shift": float(np.mean(np.abs(shift))),
                            "max_abs_shift": float(np.max(np.abs(shift))),
                            "table": sorted(tuning_rows, key=lambda r: float(r.get("objective", float("inf")))),
                            "alpha_source": "non_temp_post_tuning",
                            "tuning": {
                                "selected_alpha_override": float(best_alpha),
                                "guardrails": {
                                    "jump_q95_max": float(j95_lim),
                                    "jump_max_max": float(jmx_lim),
                                    "ann_jump_q95_max": float(a95_lim),
                                    "amp_min": float(amp_floor),
                                    "mean_shift_max": float(mean_tol),
                                },
                                "table": sorted(tuning_rows, key=lambda r: float(r.get("objective", float("inf")))),
                            },
                        }
                    else:
                        post_coupling_continuity = {
                            "enabled": False,
                            "reason": "no_better_candidate",
                            "alpha_override": None,
                            "selected_alpha": 0.0,
                            "passes": 0,
                            "before": {k: float(v) for k, v in base_cm.items()},
                            "after": {k: float(v) for k, v in base_cm.items()},
                            "table": sorted(tuning_rows, key=lambda r: float(r.get("objective", float("inf")))),
                            "alpha_source": "non_temp_post_tuning",
                        }
            else:
                post_coupling_continuity = {"enabled": False, "reason": "no_forecast_rows"}
        if isinstance(rep_extra, dict):
            rep_extra["post_coupling_continuity"] = post_coupling_continuity

        freq_tag = "monthly"
        fc_csv = fc_dir / f"{variable}_{freq_tag}_best_meta_to_{args.target_year}.csv"
        fc_pq = fc_dir / f"{variable}_{freq_tag}_best_meta_to_{args.target_year}.parquet"
        out_df.to_csv(fc_csv, index=False)
        out_df.to_parquet(fc_pq, index=False)

        chart_png = ch_dir / f"{variable}_{freq_tag}_best_meta_to_{args.target_year}.png"
        plot_forecast(out_df, variable=variable, out_png=chart_png)

        lb_csv = lb_dir / f"{variable}_meta_weights_to_{args.target_year}.csv"
        score_df.to_csv(lb_csv, index=False)

        fc_part = out_df[out_df["is_forecast"] == True].copy()
        rep = {
            "variable": variable,
            "frequency": "MS",
            "target_year": int(args.target_year),
            "walkforward_start_year": int(args.walkforward_start_year),
            "forecast_start_year": int(forecast_start_year),
            "history_points": int((out_df["is_forecast"] == False).sum()),
            "forecast_steps": int((out_df["is_forecast"] == True).sum()),
            "climate_adjustment_enabled": bool(climate_cfg.enabled),
            "climate_scenario": str(climate_cfg.scenario),
            "climate_adjustment_method": str(climate_cfg.method),
            "climate_baseline_year": float(climate_cfg.baseline_year) if np.isfinite(float(climate_cfg.baseline_year)) else None,
            "climate_temp_rate_c_per_year": float(climate_cfg.temp_rate_c_per_year),
            "humidity_per_temp_c": float(climate_cfg.humidity_per_temp_c),
            "mean_reversion_mean": float(fc_part["mean_reversion"].mean()) if not fc_part.empty else 0.0,
            "history_coverage": rep_extra.get("history_coverage", cov),
            "history_quality_flag": str(rep_extra.get("history_coverage", cov).get("quality_flag", "low")),
            "interval_inflation": float(rep_extra.get("interval_inflation", 1.0)),
            "consistency_regularization": rep_extra.get("consistency_regularization", {}),
            "continuity_smoothing": rep_extra.get("continuity_smoothing", {}),
            "post_coupling_continuity": rep_extra.get("post_coupling_continuity", {}),
            "robust_cross_member_blend": rep_extra.get("robust_cross_member_blend", {}),
            "horizon_weighting": rep_extra.get("horizon_weighting", {}),
            "pseudo_climate_delta_fallback": rep_extra.get("pseudo_climate_delta_fallback", {}),
            "force_combine_tuning": rep_extra.get("force_combine_tuning", {}),
            "temp_min_warming_rate_c_per_year": float(rep_extra.get("temp_min_warming_rate_c_per_year", 0.0)),
            "temp_trend_adjustment_mean_c": float(rep_extra.get("temp_trend_adjustment_mean_c", 0.0)),
            "temp_trend_adjustment_max_c": float(rep_extra.get("temp_trend_adjustment_max_c", 0.0)),
            "weights": rep_extra.get("weights", {}),
            "score_table": rep_extra.get("score_table", []),
            "combo_selection": rep_extra.get("combo_selection", {}),
            "combo_table": rep_extra.get("combo_table", []),
            "invalid_members": rep_extra.get("invalid_members", []),
            "rmse_proxy": float(rep_extra.get("rmse_proxy", np.nan)),
            "forecast_csv": str(fc_csv),
            "forecast_parquet": str(fc_pq),
            "chart_png": str(chart_png),
            "weights_csv": str(lb_csv),
        }
        if variable == "temp":
            rep["proxy_temp_augmentation"] = proxy_stats
            rep["coupled_temp_from_humidity"] = coupled_stats
        rep["climo_decay_member"] = climo_stats
        rep["super_ensemble_variants"] = super_ensemble_stats
        rep_json = rep_dir / f"{variable}_{freq_tag}_best_meta_report_to_{args.target_year}.json"
        rep_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

        combo_sel = rep_extra.get("combo_selection", {}) if isinstance(rep_extra, dict) else {}
        selected_models = [str(x) for x in combo_sel.get("selected_models", []) if str(x).strip()] if isinstance(combo_sel, dict) else []
        if not selected_models:
            selected_models = [str(k) for k, v in (rep_extra.get("weights", {}) or {}).items() if float(v) > 0]
        selected_model_count = (
            int(combo_sel.get("selected_model_count", len(selected_models)))
            if isinstance(combo_sel, dict)
            else int(len(selected_models))
        )

        index_rows.append(
            {
                "variable": variable,
                "frequency": "MS",
                "target_year": int(args.target_year),
                "history_points": int((out_df["is_forecast"] == False).sum()),
                "history_quality_flag": str(rep_extra.get("history_coverage", cov).get("quality_flag", "low")),
                "forecast_steps": int((out_df["is_forecast"] == True).sum()),
                "model_strategy": "best_meta_ensemble",
                "forecast_csv": str(fc_csv),
                "forecast_parquet": str(fc_pq),
                "chart_png": str(chart_png),
                "weights_csv": str(lb_csv),
                "report_json": str(rep_json),
                "input_kind": kind,
                "input_path": str(args.observations),
                "best_combo_models": ",".join(selected_models),
                "best_combo_model_count": int(selected_model_count),
                "super_ensemble_virtual_members": int(super_ensemble_stats.get("virtual_member_count", 0)),
                "horizon_weighting_enabled": bool((rep_extra.get("horizon_weighting", {}) or {}).get("enabled", False)),
                "horizon_weight_profiles": int(len((rep_extra.get("horizon_weighting", {}) or {}).get("profiles", []))),
                "pseudo_climate_delta_models": int((rep_extra.get("pseudo_climate_delta_fallback", {}) or {}).get("count", 0)),
                "robust_blend_mean_suppression": float((rep_extra.get("robust_cross_member_blend", {}) or {}).get("mean_suppression", 0.0)),
                "robust_blend_p90_suppression": float((rep_extra.get("robust_cross_member_blend", {}) or {}).get("p90_suppression", 0.0)),
                "force_combine_tuned": bool((rep_extra.get("force_combine_tuning", {}) or {}).get("enabled", False)),
                "selected_force_combine_count": int((rep_extra.get("force_combine_tuning", {}) or {}).get("selected_force_combine_count", int(max(0, args.force_combine_count)))),
                "selected_continuity_alpha_override": (
                    float((rep_extra.get("force_combine_tuning", {}) or {}).get("selected_continuity_alpha_override"))
                    if (rep_extra.get("force_combine_tuning", {}) or {}).get("selected_continuity_alpha_override") is not None
                    else np.nan
                ),
                "continuity_smoothing_enabled": bool((rep_extra.get("continuity_smoothing", {}) or {}).get("enabled", False)),
                "continuity_selected_alpha": float((rep_extra.get("continuity_smoothing", {}) or {}).get("selected_alpha", 0.0)),
                "post_coupling_continuity_enabled": bool((rep_extra.get("post_coupling_continuity", {}) or {}).get("enabled", False)),
                "post_coupling_continuity_alpha": float((rep_extra.get("post_coupling_continuity", {}) or {}).get("selected_alpha", 0.0)),
            }
        )
        if variable == "temp":
            index_rows[-1]["proxy_temp_augmentation_enabled"] = bool(proxy_stats.get("enabled", False))
            index_rows[-1]["proxy_temp_added_points"] = int(proxy_stats.get("added_points", 0))
        out_map[variable] = out_df

    cov_df = pd.DataFrame(coverage_rows).sort_values("variable") if coverage_rows else pd.DataFrame()
    cov_csv = out / "history_coverage_summary.csv"
    cov_md = out / "history_coverage_summary.md"
    cov_df.to_csv(cov_csv, index=False)
    cov_md.write_text("# History Coverage Summary\n\n" + cov_df.round(4).to_markdown(index=False) + "\n", encoding="utf-8")

    idx = pd.DataFrame(index_rows).sort_values("variable") if index_rows else pd.DataFrame()
    idx_csv = out / f"best_meta_index_to_{args.target_year}.csv"
    idx_pq = out / f"best_meta_index_to_{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    write_annual_compare(
        out=out,
        temp_fc=out_map.get("temp"),
        hum_fc=out_map.get("humidity"),
        start_year=int(forecast_start_year),
        target_year=int(args.target_year),
    )
    write_annual_compare_all_years(
        out=out,
        temp_df=out_map.get("temp"),
        hum_df=out_map.get("humidity"),
    )
    write_annual_pairwise_correlations(
        out=out,
        out_map=out_map,
        forecast_start_year=int(forecast_start_year),
        target_year=int(args.target_year),
    )

    print("Best meta ensemble completed.")
    print(f"Index: {idx_csv}")
    if not idx.empty:
        print(idx[["variable", "frequency", "forecast_steps", "forecast_csv"]].to_string(index=False))


if __name__ == "__main__":
    main()
