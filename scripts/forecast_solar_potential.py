#!/usr/bin/env python3
"""Advanced solar potential estimation from climate forecasts.

Inputs:
- temperature
- humidity
- precipitation
- pressure

Pipeline:
1) Select best available climate source bundle (auto/forecast/clean2035/custom).
2) Merge climate series on timestamp and compute solar geometry features.
3) Build a physics-informed solar potential baseline.
4) Calibrate baseline with observed solar data (bias-only or linear, when available).
5) Optionally apply ML correction (if enough historical target overlap exists).
6) Estimate uncertainty (Monte Carlo for heuristic mode; normal approx for ML mode).
7) Export forecast table and diagnostics report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

_MPL_DIR = Path("/tmp/mplconfig_forecast_solar")
try:
    _MPL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
except Exception:
    pass

try:
    import matplotlib.pyplot as plt

    MPL_OK = True
except Exception:
    MPL_OK = False

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


DEFAULT_FORECAST_ROOT = Path("/Users/yasinkaya/Hackhaton/output/forecast_package/forecasts")
DEFAULT_CLEAN_ROOT = Path("/Users/yasinkaya/Hackhaton/output/forecast_package/clean_2035_datasets")
DEFAULT_SOLAR_OBS = Path("/Users/yasinkaya/Hackhaton/output/quant_all_visuals_input/visual_measurements_all.csv")
DEFAULT_CHART_ROOT = Path("/Users/yasinkaya/Hackhaton/output/forecast_package/charts")
DEFAULT_EXTERNAL_ROOT = Path("/Users/yasinkaya/Hackhaton/output/forecast_package/external")

DEFAULT_FORECAST_FILES = {
    "temp": DEFAULT_FORECAST_ROOT / "temp_monthly_forecast.parquet",
    "humidity": DEFAULT_FORECAST_ROOT / "humidity_monthly_forecast.parquet",
    "precip": DEFAULT_FORECAST_ROOT / "precip_monthly_forecast.parquet",
    "pressure": DEFAULT_FORECAST_ROOT / "pressure_monthly_forecast.parquet",
}

DEFAULT_CLEAN_FILES = {
    "temp": DEFAULT_CLEAN_ROOT / "temp_monthly_continuous_to_2035.csv",
    "humidity": DEFAULT_CLEAN_ROOT / "humidity_monthly_continuous_to_2035.csv",
    "precip": DEFAULT_CLEAN_ROOT / "precip_monthly_continuous_to_2035.csv",
    "pressure": DEFAULT_CLEAN_ROOT / "pressure_monthly_continuous_to_2035.csv",
}

SIGMA_MIN = {"temp": 0.35, "humidity": 1.50, "precip": 2.00, "pressure": 0.80}
SIGMA_FRAC_STD = {"temp": 0.08, "humidity": 0.06, "precip": 0.12, "pressure": 0.05}
PV_GAMMA_MAP = {
    "standard": -0.0047,  # PVWatts V5 table (approx -0.47%/C)
    "premium": -0.0035,   # approx -0.35%/C
    "thin_film": -0.0020, # approx -0.20%/C
}


@dataclass
class CalibrationInfo:
    enabled: bool
    points: int
    slope: float
    intercept: float
    mae: float
    r2: float
    method: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advanced solar potential forecast from temp/humidity/precip/pressure."
    )
    p.add_argument("--temp-forecast", type=Path, default=DEFAULT_FORECAST_FILES["temp"])
    p.add_argument("--humidity-forecast", type=Path, default=DEFAULT_FORECAST_FILES["humidity"])
    p.add_argument("--precip-forecast", type=Path, default=DEFAULT_FORECAST_FILES["precip"])
    p.add_argument("--pressure-forecast", type=Path, default=DEFAULT_FORECAST_FILES["pressure"])
    p.add_argument("--solar-observations", type=Path, default=DEFAULT_SOLAR_OBS, help="Optional observed solar file.")
    p.add_argument(
        "--source-mode",
        type=str,
        default="auto",
        choices=["auto", "forecast", "clean2035", "custom"],
        help="Source bundle selection. auto compares forecast vs clean2035 and picks broader overlap.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "heuristic", "ml"],
        help="auto: use ML if enough target data, otherwise heuristic.",
    )
    p.add_argument("--latitude", type=float, default=41.01, help="Station latitude in degrees.")
    p.add_argument("--longitude", type=float, default=28.95, help="Station longitude in degrees.")
    p.add_argument("--elevation-m", type=float, default=39.0, help="Station elevation above sea level [m].")
    p.add_argument(
        "--internet-extra-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "openmeteo"],
        help="Fetch extra monthly predictors from internet (Open-Meteo climate API).",
    )
    p.add_argument(
        "--openmeteo-model",
        type=str,
        default="EC_Earth3P_HR",
        help="Open-Meteo climate model identifier.",
    )
    p.add_argument(
        "--openmeteo-timezone",
        type=str,
        default="Europe/Istanbul",
        help="Timezone passed to Open-Meteo climate API.",
    )
    p.add_argument(
        "--openmeteo-cache-csv",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT / "openmeteo_monthly_extras.csv",
        help="Cache file for downloaded Open-Meteo monthly extras.",
    )
    p.add_argument(
        "--refresh-internet-extra",
        type=str,
        default="false",
        help="true/false: refresh internet extra cache before run.",
    )
    p.add_argument(
        "--solar-reference-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "nasa_power"],
        help="Reference solar source when local observations are sparse (auto/none/nasa_power).",
    )
    p.add_argument(
        "--nasa-power-cache-csv",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT / "nasa_power_monthly_solar.csv",
        help="Cache file for NASA POWER monthly solar reference.",
    )
    p.add_argument(
        "--refresh-solar-reference",
        type=str,
        default="false",
        help="true/false: refresh NASA POWER cache before run.",
    )
    p.add_argument("--internet-timeout-seconds", type=int, default=45, help="HTTP timeout for internet calls.")
    p.add_argument("--temp-opt-c", type=float, default=23.0, help="Temperature optimum for PV performance.")
    p.add_argument("--temp-penalty-per-c", type=float, default=0.004, help="Efficiency penalty per C above optimum.")
    p.add_argument(
        "--forecast-temp-internet-blend",
        type=float,
        default=0.90,
        help="Forecast-only blend weight for internet temperature (0 local-only, 1 internet-only).",
    )
    p.add_argument(
        "--forecast-humidity-internet-blend",
        type=float,
        default=0.65,
        help="Forecast-only blend weight for internet humidity (%).",
    )
    p.add_argument(
        "--forecast-precip-internet-blend",
        type=float,
        default=0.55,
        help="Forecast-only blend weight for internet precipitation proxy (mm/day).",
    )
    p.add_argument(
        "--forecast-pressure-internet-blend",
        type=float,
        default=0.60,
        help="Forecast-only blend weight for internet surface pressure.",
    )
    p.add_argument(
        "--forecast-shortwave-internet-blend",
        type=float,
        default=0.00,
        help="Forecast-only blend weight to nudge solar potential by internet shortwave-based proxy.",
    )
    p.add_argument(
        "--forecast-cloudiness-internet-blend",
        type=float,
        default=0.70,
        help="Forecast-only blend weight to align cloudiness output with internet cloud cover.",
    )
    p.add_argument(
        "--assimilation-lead-decay-half-life-months",
        type=float,
        default=0.0,
        help="Half-life (months) for forecast lead-time decay of internet assimilation weights; <=0 disables decay.",
    )
    p.add_argument(
        "--assimilation-lead-decay-min-factor",
        type=float,
        default=0.35,
        help="Minimum long-horizon factor for internet assimilation weight decay (0-1).",
    )
    p.add_argument(
        "--pv-module-type",
        type=str,
        default="standard",
        choices=["standard", "premium", "thin_film"],
        help="PV module family for temperature coefficient defaults (PVWatts V5 style).",
    )
    p.add_argument(
        "--gamma-pdc",
        type=float,
        default=float("nan"),
        help="PV power temperature coefficient [1/C]. If NaN, derived from --pv-module-type.",
    )
    p.add_argument("--clip-low-q", type=float, default=0.10)
    p.add_argument("--clip-high-q", type=float, default=0.90)
    p.add_argument("--ml-min-months", type=int, default=24, help="Minimum overlapping solar months to enable ML.")
    p.add_argument("--ml-holdout", type=int, default=24, help="Holdout size for ML validation.")
    p.add_argument(
        "--min-calibration-points",
        type=int,
        default=6,
        help="Minimum overlapping solar months required to apply calibration.",
    )
    p.add_argument(
        "--low-point-calibration-shrink",
        type=float,
        default=0.35,
        help="Shrink factor for 1-2 point bias calibration (0-1). Lower means safer calibration.",
    )
    p.add_argument("--mc-samples", type=int, default=400, help="Monte Carlo sample count for uncertainty.")
    p.add_argument(
        "--uncertainty-z",
        type=float,
        default=1.2815515655446004,
        help="z-score used when converting variable low/high bands into sigma (default ~80%% interval).",
    )
    p.add_argument(
        "--forecast-smoothing-alpha",
        type=float,
        default=0.22,
        help="Exponential smoothing factor for forecast period only (0 disables).",
    )
    p.add_argument(
        "--horizon-aware",
        type=str,
        default="true",
        help="true/false: apply lead-time aware blending and uncertainty scaling.",
    )
    p.add_argument("--horizon-h1-months", type=int, default=12, help="Short-horizon limit in months.")
    p.add_argument("--horizon-h2-months", type=int, default=36, help="Medium-horizon limit in months.")
    p.add_argument(
        "--horizon-mid-blend-max",
        type=float,
        default=0.20,
        help="Max climatology blend weight by end of medium horizon.",
    )
    p.add_argument(
        "--horizon-long-blend",
        type=float,
        default=0.25,
        help="Base climatology blend weight after medium horizon.",
    )
    p.add_argument(
        "--horizon-long-blend-growth-per-year",
        type=float,
        default=0.02,
        help="Additional long-horizon blend increase per year.",
    )
    p.add_argument(
        "--horizon-blend-max",
        type=float,
        default=0.70,
        help="Upper bound for horizon climatology blend weight.",
    )
    p.add_argument(
        "--min-history-for-horizon-blend",
        type=int,
        default=12,
        help="Minimum historical months required before applying horizon-aware blending.",
    )
    p.add_argument(
        "--uncertainty-growth-per-year",
        type=float,
        default=0.04,
        help="Relative uncertainty growth per forecast year in horizon-aware mode.",
    )
    p.add_argument(
        "--scenario-enable",
        type=str,
        default="true",
        help="true/false: generate one autocorrelated stochastic realization for forecast months.",
    )
    p.add_argument(
        "--scenario-ar1-rho",
        type=float,
        default=0.55,
        help="AR(1) persistence for stochastic forecast realization (-0.95 to 0.95).",
    )
    p.add_argument(
        "--scenario-scale",
        type=float,
        default=0.65,
        help="Scale of realization volatility relative to uncertainty sigma.",
    )
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_FORECAST_ROOT / "solar_potential_monthly_forecast.csv")
    p.add_argument("--output-parquet", type=Path, default=DEFAULT_FORECAST_ROOT / "solar_potential_monthly_forecast.parquet")
    p.add_argument("--report-json", type=Path, default=DEFAULT_FORECAST_ROOT / "solar_potential_report.json")
    p.add_argument(
        "--output-chart",
        type=Path,
        default=DEFAULT_CHART_ROOT / "solar_potential_monthly_to_2035_pro.png",
        help="Main solar potential chart path (PNG).",
    )
    p.add_argument(
        "--output-driver-chart",
        type=Path,
        default=DEFAULT_CHART_ROOT / "solar_driver_panels.png",
        help="Driver panel chart path (PNG).",
    )
    p.add_argument(
        "--export-separated",
        type=str,
        default="true",
        help="true/false: export separated per-variable CSV and charts.",
    )
    p.add_argument(
        "--separated-output-dir",
        type=Path,
        default=DEFAULT_FORECAST_ROOT / "separated",
        help="Directory for separated per-variable CSV outputs.",
    )
    p.add_argument(
        "--separated-chart-dir",
        type=Path,
        default=DEFAULT_CHART_ROOT / "separated",
        help="Directory for separated per-variable chart outputs.",
    )
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise SystemExit(f"Unsupported file extension: {path}")


def pick_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def normalize_bool(series: pd.Series) -> pd.Series:
    t = series.astype(str).str.strip().str.lower()
    return t.isin({"1", "true", "yes", "y", "on"})


def to_bool_text(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_forecast(path: Path, variable: str) -> pd.DataFrame:
    raw = read_table(path)
    ts_col = pick_column(raw, ["timestamp", "ds", "date", "datetime", "time", "tarih"])
    val_col = pick_column(raw, ["yhat", "value", "forecast", "prediction", variable])
    low_col = pick_column(raw, ["low", "yhat_lower", "lower", "p10", "q10"])
    high_col = pick_column(raw, ["high", "yhat_upper", "upper", "p90", "q90"])
    fc_col = pick_column(raw, ["is_forecast", "forecast_flag"])
    if ts_col is None:
        raise SystemExit(f"{path}: cannot find timestamp column")
    if val_col is None:
        raise SystemExit(f"{path}: cannot find forecast value column")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[ts_col], errors="coerce"),
            variable: pd.to_numeric(raw[val_col], errors="coerce"),
            f"{variable}_low": pd.to_numeric(raw[low_col], errors="coerce") if low_col is not None else np.nan,
            f"{variable}_high": pd.to_numeric(raw[high_col], errors="coerce") if high_col is not None else np.nan,
            f"is_forecast_{variable}": normalize_bool(raw[fc_col]) if fc_col is not None else False,
        }
    )
    out = out.dropna(subset=["timestamp", variable]).sort_values("timestamp")
    if out.empty:
        raise SystemExit(f"{path}: no usable rows after parsing")
    out = (
        out.groupby("timestamp", as_index=False)
        .agg(
            {
                variable: "mean",
                f"{variable}_low": "mean",
                f"{variable}_high": "mean",
                f"is_forecast_{variable}": "max",
            }
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out


def source_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "temp": args.temp_forecast,
        "humidity": args.humidity_forecast,
        "precip": args.precip_forecast,
        "pressure": args.pressure_forecast,
    }


def load_source_bundle(paths: dict[str, Path], label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    vars_ = ["temp", "humidity", "precip", "pressure"]
    tables: dict[str, pd.DataFrame] = {}
    per_var: dict[str, Any] = {}

    for var in vars_:
        p = paths[var]
        t = load_forecast(p, var)
        tables[var] = t
        per_var[var] = {
            "path": str(p),
            "rows": int(len(t)),
            "start": str(pd.to_datetime(t["timestamp"]).min().date()),
            "end": str(pd.to_datetime(t["timestamp"]).max().date()),
        }

    merged = tables["temp"]
    for var in ["humidity", "precip", "pressure"]:
        merged = merged.merge(tables[var], on="timestamp", how="inner")

    if merged.empty:
        raise SystemExit(f"Source bundle '{label}' has no overlapping timestamps across variables.")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    fc_cols = [f"is_forecast_{v}" for v in vars_]
    merged["is_forecast"] = merged[fc_cols].any(axis=1)

    meta: dict[str, Any] = {
        "selected_label": label,
        "paths": {k: str(v) for k, v in paths.items()},
        "rows": int(len(merged)),
        "history_rows": int((~merged["is_forecast"]).sum()),
        "forecast_rows": int(merged["is_forecast"].sum()),
        "start": str(pd.to_datetime(merged["timestamp"]).min().date()),
        "end": str(pd.to_datetime(merged["timestamp"]).max().date()),
        "per_variable": per_var,
    }
    return merged, meta


def select_source_and_load(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(args.source_mode).strip().lower()
    custom_paths = source_paths_from_args(args)
    clean_paths = DEFAULT_CLEAN_FILES.copy()

    if mode == "forecast" or mode == "custom":
        merged, meta = load_source_bundle(custom_paths, label=mode)
        return merged, meta

    if mode == "clean2035":
        merged, meta = load_source_bundle(clean_paths, label="clean2035")
        return merged, meta

    # auto mode
    candidates: list[tuple[tuple[int, int], str, pd.DataFrame, dict[str, Any]]] = []
    errors: dict[str, str] = {}

    for label, paths in [("forecast", custom_paths), ("clean2035", clean_paths)]:
        try:
            merged, meta = load_source_bundle(paths, label=label)
            score = (int(meta["rows"]), int(meta["history_rows"]))
            candidates.append((score, label, merged, meta))
        except Exception as exc:
            errors[label] = str(exc)

    if not candidates:
        msg = "; ".join(f"{k}:{v}" for k, v in errors.items()) if errors else "unknown"
        raise SystemExit(f"Could not load any source bundle in auto mode. {msg}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, _, merged, meta = candidates[0]
    meta["auto_candidates"] = [
        {
            "label": label,
            "rows": int(m["rows"]),
            "history_rows": int(m["history_rows"]),
            "start": m["start"],
            "end": m["end"],
        }
        for _, label, _, m in candidates
    ]
    if errors:
        meta["auto_errors"] = errors
    return merged, meta


def fetch_openmeteo_monthly_extras(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    model: str,
    timezone: str,
    timeout_seconds: int,
    cache_csv: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_csv = Path(cache_csv)
    meta: dict[str, Any] = {
        "provider": "open-meteo",
        "model": str(model),
        "timezone": str(timezone),
        "cache_csv": str(cache_csv),
        "used_cache": False,
        "download_url": None,
        "download_transport": None,
    }
    required_cols = {
        "timestamp",
        "cloud_cover_internet_pct",
        "wind_speed_internet_kmh",
        "shortwave_internet_kwh_m2_day",
        "temp_internet_c",
        "humidity_internet_pct",
        "precip_internet_mm_day",
        "pressure_internet",
    }
    signal_cols = [
        "cloud_cover_internet_pct",
        "wind_speed_internet_kmh",
        "shortwave_internet_kwh_m2_day",
        "temp_internet_c",
        "humidity_internet_pct",
        "precip_internet_mm_day",
        "pressure_internet",
    ]

    if cache_csv.exists() and (not refresh):
        try:
            cached = pd.read_csv(cache_csv, parse_dates=["timestamp"])
            cols = set(cached.columns)
            if "timestamp" in cols:
                for c in (required_cols - cols):
                    if c != "timestamp":
                        cached[c] = np.nan
                cached["timestamp"] = pd.to_datetime(cached["timestamp"], errors="coerce")
                cached = cached.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                cache_complete = bool(required_cols.issubset(cols))

                start_ts = pd.to_datetime(start_date).to_period("M").to_timestamp()
                end_ts = pd.to_datetime(end_date).to_period("M").to_timestamp()
                expected_rows = int(len(pd.date_range(start=start_ts, end=end_ts, freq="MS")))
                period = cached[(cached["timestamp"] >= start_ts) & (cached["timestamp"] <= end_ts)].copy()
                period_rows = int(len(period))
                window_coverage = float(period_rows / max(1, expected_rows))

                signal_coverage = {}
                for c in signal_cols:
                    signal_coverage[c] = (
                        float(pd.to_numeric(period[c], errors="coerce").notna().mean()) if period_rows else 0.0
                    )
                coverage_ok = bool(window_coverage >= 0.95 and all(v >= 0.95 for v in signal_coverage.values()))

                if cache_complete and coverage_ok:
                    meta["used_cache"] = True
                    meta["rows"] = int(len(cached))
                    meta["cache_partial"] = False
                    meta["cache_window_coverage"] = window_coverage
                    meta["cache_signal_coverage"] = signal_coverage
                    return cached, meta

                meta["cache_partial"] = bool(not cache_complete)
                meta["cache_window_coverage"] = window_coverage
                meta["cache_signal_coverage"] = signal_coverage
                meta["cache_refresh_reason"] = "stale_or_incomplete_cache"
        except Exception:
            pass

    params = {
        "latitude": f"{float(latitude):.4f}",
        "longitude": f"{float(longitude):.4f}",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "models": str(model),
        "daily": (
            "cloud_cover_mean,wind_speed_10m_mean,shortwave_radiation_sum,"
            "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,pressure_msl_mean,surface_pressure_mean"
        ),
        "timezone": str(timezone),
    }
    url = "https://climate-api.open-meteo.com/v1/climate?" + urlencode(params)
    meta["download_url"] = url

    payload: dict[str, Any] | None = None
    try:
        with urlopen(url, timeout=max(5, int(timeout_seconds))) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        meta["download_transport"] = "urllib"
    except Exception as exc:
        try:
            proc = subprocess.run(
                ["curl", "-sS", url],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(10, int(timeout_seconds) + 5),
            )
            payload = json.loads(proc.stdout)
            meta["download_transport"] = "curl_fallback"
        except Exception as exc2:
            raise RuntimeError(f"network_error:{exc}; curl_fallback_error:{exc2}") from exc2

    assert payload is not None
    daily = payload.get("daily") or {}
    t = daily.get("time")
    cc = daily.get("cloud_cover_mean")
    ws = daily.get("wind_speed_10m_mean")
    sw = daily.get("shortwave_radiation_sum")
    tp = daily.get("temperature_2m_mean")
    rh = daily.get("relative_humidity_2m_mean")
    pr = daily.get("precipitation_sum")
    ps_msl = daily.get("pressure_msl_mean")
    ps_sfc = daily.get("surface_pressure_mean")
    ps = ps_msl if ps_msl is not None else ps_sfc
    if t is None or cc is None or ws is None or sw is None or tp is None or rh is None or pr is None or ps is None:
        raise RuntimeError("Open-Meteo response missing required daily fields.")

    n = len(t)
    if not (len(cc) == len(ws) == len(sw) == len(tp) == len(rh) == len(pr) == len(ps) == n):
        raise RuntimeError("Open-Meteo response length mismatch.")

    daily_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(t, errors="coerce"),
            "cloud_cover_internet_pct": pd.to_numeric(cc, errors="coerce"),
            "wind_speed_internet_kmh": pd.to_numeric(ws, errors="coerce"),
            # API field is MJ/m2 daily sum; convert to kWh/m2/day for consistency.
            "shortwave_internet_kwh_m2_day": pd.to_numeric(sw, errors="coerce") * 0.2777777778,
            "temp_internet_c": pd.to_numeric(tp, errors="coerce"),
            "humidity_internet_pct": pd.to_numeric(rh, errors="coerce"),
            # Daily precipitation sum already in mm/day.
            "precip_internet_mm_day": pd.to_numeric(pr, errors="coerce"),
            "pressure_internet": pd.to_numeric(ps, errors="coerce"),
        }
    ).dropna(subset=["timestamp"])

    out = (
        daily_df.set_index("timestamp")
        .resample("MS")
        .mean()
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    for c in required_cols - {"timestamp"}:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_csv, index=False)
    meta["rows"] = int(len(out))
    return out, meta


def attach_internet_extras(merged: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(args.internet_extra_mode).strip().lower()
    if mode == "none":
        return merged, {"enabled": False, "mode": "none"}

    ts = pd.to_datetime(merged["timestamp"])
    start_date = str(ts.min().date())
    end_date = str(ts.max().date())
    refresh = bool(to_bool_text(args.refresh_internet_extra))
    meta: dict[str, Any] = {
        "enabled": False,
        "mode": mode,
        "start_date": start_date,
        "end_date": end_date,
        "refresh": refresh,
        "status": "not_started",
    }

    try:
        extra, fetch_meta = fetch_openmeteo_monthly_extras(
            latitude=float(args.latitude),
            longitude=float(args.longitude),
            start_date=start_date,
            end_date=end_date,
            model=str(args.openmeteo_model),
            timezone=str(args.openmeteo_timezone),
            timeout_seconds=int(args.internet_timeout_seconds),
            cache_csv=Path(args.openmeteo_cache_csv),
            refresh=refresh,
        )
        merged2 = merged.merge(extra, on="timestamp", how="left")
        extra_cols = [
            "cloud_cover_internet_pct",
            "wind_speed_internet_kmh",
            "shortwave_internet_kwh_m2_day",
            "temp_internet_c",
            "humidity_internet_pct",
            "precip_internet_mm_day",
            "pressure_internet",
        ]
        for c in extra_cols:
            if c not in merged2.columns:
                merged2[c] = np.nan
            merged2[c] = pd.to_numeric(merged2[c], errors="coerce").interpolate(method="linear").ffill().bfill()
            merged2[c] = merged2[c].astype(float)

        # Convert precipitation mean daily depth to monthly total (mm/month)
        # to match local monthly precipitation scale.
        if "precip_internet_mm_day" in merged2.columns:
            dim = pd.to_datetime(merged2["timestamp"]).dt.days_in_month.astype(float)
            merged2["precip_internet_mm_month"] = merged2["precip_internet_mm_day"] * dim
        else:
            merged2["precip_internet_mm_month"] = np.nan

        coverage_cols = extra_cols + ["precip_internet_mm_month"]
        coverage = {c: float(np.isfinite(merged2[c]).mean()) for c in coverage_cols}
        meta.update(
            {
                "enabled": True,
                "status": "ok",
                "coverage": coverage,
                "fetch": fetch_meta,
            }
        )
        return merged2, meta
    except Exception as exc:
        meta["status"] = f"failed:{exc}"
        if mode == "openmeteo":
            raise SystemExit(f"Internet extra fetch failed: {exc}")
        return merged, meta


def apply_internet_assimilation(merged: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = merged.copy()
    fc = out["is_forecast"].astype(bool)
    meta: dict[str, Any] = {"enabled": False, "reason": "no_variable_applied", "variables": {}}
    any_applied = False
    ts_all = pd.to_datetime(out["timestamp"], errors="coerce")
    half_life_m = float(args.assimilation_lead_decay_half_life_months)
    decay_min = float(np.clip(float(args.assimilation_lead_decay_min_factor), 0.0, 1.0))
    lead_decay = pd.Series(np.where(fc.to_numpy(), 1.0, 0.0), index=out.index, dtype=float)
    if fc.any() and (half_life_m > 0):
        first_fc_ts = ts_all[fc].min()
        lead_months = (
            (ts_all.dt.year - int(first_fc_ts.year)) * 12 + (ts_all.dt.month - int(first_fc_ts.month))
        ).astype(float)
        lead_months = np.clip(lead_months, 0.0, None)
        decay = decay_min + (1.0 - decay_min) * np.exp(-np.log(2.0) * lead_months / max(1e-6, half_life_m))
        lead_decay = pd.Series(np.where(fc.to_numpy(), decay, 0.0), index=out.index, dtype=float)
    meta["lead_decay"] = {
        "enabled": bool(half_life_m > 0),
        "half_life_months": float(half_life_m),
        "min_factor": float(decay_min),
        "forecast_decay_mean": float(lead_decay[fc].mean()) if fc.any() else None,
        "forecast_decay_p10": float(lead_decay[fc].quantile(0.10)) if fc.any() else None,
        "forecast_decay_p90": float(lead_decay[fc].quantile(0.90)) if fc.any() else None,
    }

    configs = [
        ("temp", "temp_internet_c", "forecast_temp_internet_blend", -60.0, 70.0, "c"),
        ("humidity", "humidity_internet_pct", "forecast_humidity_internet_blend", 0.0, 100.0, "pct"),
        ("precip", "precip_internet_mm_month", "forecast_precip_internet_blend", 0.0, float("inf"), "mm"),
        ("pressure", "pressure_internet", "forecast_pressure_internet_blend", 0.0, float("inf"), "raw"),
    ]
    min_corr_floor = {
        "temp": 0.55,
        "humidity": 0.10,
        "precip": 0.08,
        "pressure": 0.30,
    }
    min_weight_fraction = {
        "temp": 0.0,
        "humidity": 1.0,
        "precip": 1.0,
        "pressure": 0.0,
    }

    def choose_overlap_mask(
        local_v: pd.Series,
        inet_v: pd.Series,
        hist_mask: pd.Series,
        *,
        min_points: int = 10,
        min_history_points: int = 24,
    ) -> tuple[pd.Series, str]:
        hist = pd.Series(hist_mask, index=local_v.index).astype(bool)
        valid = local_v.notna() & inet_v.notna()
        m_hist = hist & valid
        m_all = valid
        if int(m_hist.sum()) >= int(max(min_history_points, min_points)):
            return m_hist, "history_overlap"
        if int(m_all.sum()) >= int(max(6, min_points)):
            return m_all, "all_overlap_fallback"
        return m_hist, "insufficient_overlap"

    def harmonize_internet_series(
        local_s: pd.Series,
        inet_s: pd.Series,
        hist_mask: pd.Series,
        *,
        min_points: int = 10,
        min_history_points: int = 24,
        min_corr: float = 0.35,
    ) -> tuple[pd.Series, dict[str, Any]]:
        local_v = pd.to_numeric(local_s, errors="coerce").astype(float)
        inet_v = pd.to_numeric(inet_s, errors="coerce").astype(float)
        m, basis = choose_overlap_mask(
            local_v=local_v,
            inet_v=inet_v,
            hist_mask=hist_mask,
            min_points=min_points,
            min_history_points=min_history_points,
        )

        info: dict[str, Any] = {
            "attempted": True,
            "applied": False,
            "reason": "insufficient_history_overlap",
            "points": int(m.sum()),
            "basis": basis,
            "scale_ratio": None,
            "corr_history": None,
            "raw_mae": None,
            "aligned_mae": None,
            "slope": None,
            "intercept": None,
            "corr_effective": None,
        }
        if int(m.sum()) < int(max(6, min_points)):
            return inet_v, info

        x = inet_v[m].to_numpy(dtype=float)
        y = local_v[m].to_numpy(dtype=float)
        x_med = float(np.nanmedian(np.abs(x)))
        y_med = float(np.nanmedian(np.abs(y)))
        scale_ratio = float(x_med / max(1e-9, y_med))
        raw_mae = float(np.mean(np.abs(y - x)))
        need_alignment = bool((scale_ratio < 0.50) or (scale_ratio > 2.00))
        info["scale_ratio"] = scale_ratio
        info["raw_mae"] = raw_mae
        if not need_alignment:
            if (np.std(x) > 1e-9) and (np.std(y) > 1e-9):
                corr_eff = float(np.corrcoef(x, y)[0, 1])
                info["corr_effective"] = corr_eff if np.isfinite(corr_eff) else None
            info["reason"] = "already_harmonized"
            return inet_v, info

        if (np.std(x) <= 1e-9) or (np.std(y) <= 1e-9):
            info["reason"] = "low_variance"
            return inet_v, info
        corr = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(corr):
            corr = None
        info["corr_history"] = corr
        if (corr is None) or (corr < float(min_corr)):
            info["reason"] = "low_correlation_for_alignment"
            return inet_v, info

        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)
        if (not np.isfinite(slope)) or (not np.isfinite(intercept)):
            info["reason"] = "fit_failed_non_finite"
            return inet_v, info
        y_hat = slope * x + intercept
        aligned_mae = float(np.mean(np.abs(y - y_hat)))
        info["aligned_mae"] = aligned_mae
        info["slope"] = slope
        info["intercept"] = intercept
        if aligned_mae > raw_mae * 0.98:
            info["reason"] = "fit_no_gain"
            return inet_v, info

        y_lo = float(np.nanpercentile(y, 1.0))
        y_hi = float(np.nanpercentile(y, 99.0))
        pad = 0.20 * max(1.0, y_hi - y_lo)
        aligned_full = pd.Series(np.clip(slope * inet_v + intercept, y_lo - pad, y_hi + pad), index=inet_v.index)
        y_eff = aligned_full[m].to_numpy(dtype=float)
        if (np.std(y_eff) > 1e-9) and (np.std(y) > 1e-9):
            corr_eff = float(np.corrcoef(y_eff, y)[0, 1])
            info["corr_effective"] = corr_eff if np.isfinite(corr_eff) else None
        info["applied"] = True
        info["reason"] = "applied"
        return aligned_full.astype(float), info

    def seasonal_bias_correct(
        local_s: pd.Series,
        inet_s: pd.Series,
        ts_s: pd.Series,
        hist_mask: pd.Series,
        *,
        min_points: int = 10,
        min_points_per_month: int = 3,
        min_history_points: int = 24,
    ) -> tuple[pd.Series, dict[str, Any]]:
        local_v = pd.to_numeric(local_s, errors="coerce").astype(float)
        inet_v = pd.to_numeric(inet_s, errors="coerce").astype(float)
        ts = pd.to_datetime(ts_s, errors="coerce")
        m, basis = choose_overlap_mask(
            local_v=local_v,
            inet_v=inet_v,
            hist_mask=hist_mask,
            min_points=min_points,
            min_history_points=min_history_points,
        )
        info: dict[str, Any] = {
            "attempted": True,
            "applied": False,
            "reason": "insufficient_overlap",
            "points": int(m.sum()),
            "basis": basis,
            "global_bias": None,
            "mae_before": None,
            "mae_after": None,
            "months_with_specific_bias": 0,
        }
        if int(m.sum()) < int(max(6, min_points)):
            return inet_v, info

        diff = local_v[m] - inet_v[m]
        global_bias = float(np.nanmedian(diff))
        month = ts.dt.month.astype("Int64")
        by_m = pd.DataFrame({"month": month[m], "diff": diff}).dropna()
        month_stats = by_m.groupby("month")["diff"].agg(["median", "count"]).reset_index()
        month_bias_map = {int(r["month"]): float(r["median"]) for _, r in month_stats.iterrows() if int(r["count"]) >= int(min_points_per_month)}
        bias_series = month.map(lambda x: month_bias_map.get(int(x), global_bias) if pd.notna(x) else global_bias).astype(float)
        corrected = inet_v + bias_series

        mae_before = float(np.mean(np.abs(local_v[m] - inet_v[m])))
        mae_after = float(np.mean(np.abs(local_v[m] - corrected[m])))
        info["global_bias"] = global_bias
        info["mae_before"] = mae_before
        info["mae_after"] = mae_after
        info["months_with_specific_bias"] = int(len(month_bias_map))

        if mae_after > mae_before * 1.05:
            info["reason"] = "mae_degraded"
            return inet_v, info

        info["applied"] = True
        info["reason"] = "applied"
        return corrected.astype(float), info

    def quantile_map_correct(
        local_s: pd.Series,
        inet_s: pd.Series,
        hist_mask: pd.Series,
        *,
        min_points: int = 36,
        min_history_points: int = 48,
        q_count: int = 31,
        min_corr: float = 0.20,
    ) -> tuple[pd.Series, dict[str, Any]]:
        local_v = pd.to_numeric(local_s, errors="coerce").astype(float)
        inet_v = pd.to_numeric(inet_s, errors="coerce").astype(float)
        m, basis = choose_overlap_mask(
            local_v=local_v,
            inet_v=inet_v,
            hist_mask=hist_mask,
            min_points=min_points,
            min_history_points=min_history_points,
        )
        info: dict[str, Any] = {
            "attempted": True,
            "applied": False,
            "reason": "insufficient_overlap",
            "points": int(m.sum()),
            "basis": basis,
            "mae_before": None,
            "mae_after": None,
            "corr_before": None,
            "corr_after": None,
            "q_count": int(q_count),
        }
        if int(m.sum()) < int(max(12, min_points)):
            return inet_v, info

        x = inet_v[m].to_numpy(dtype=float)
        y = local_v[m].to_numpy(dtype=float)
        if (np.std(x) <= 1e-9) or (np.std(y) <= 1e-9):
            info["reason"] = "low_variance"
            return inet_v, info
        corr_before = float(np.corrcoef(x, y)[0, 1])
        corr_before = corr_before if np.isfinite(corr_before) else None
        info["corr_before"] = corr_before
        if (corr_before is None) or (corr_before < float(min_corr)):
            info["reason"] = "low_correlation_for_quantile_map"
            return inet_v, info

        qn = int(max(11, min(101, q_count)))
        q = np.linspace(0.01, 0.99, qn)
        xq = np.quantile(x, q)
        yq = np.quantile(y, q)
        # Ensure strictly increasing x-grid for interpolation.
        xq_u, idx_u = np.unique(xq, return_index=True)
        yq_u = yq[idx_u]
        if len(xq_u) < 8:
            info["reason"] = "insufficient_unique_quantiles"
            return inet_v, info

        mapped_full = pd.Series(
            np.interp(
                inet_v.to_numpy(dtype=float),
                xq_u.astype(float),
                yq_u.astype(float),
                left=float(yq_u[0]),
                right=float(yq_u[-1]),
            ),
            index=inet_v.index,
            dtype=float,
        )

        y_hat = mapped_full[m].to_numpy(dtype=float)
        mae_before = float(np.mean(np.abs(y - x)))
        mae_after = float(np.mean(np.abs(y - y_hat)))
        corr_after = float(np.corrcoef(y_hat, y)[0, 1]) if (np.std(y_hat) > 1e-9 and np.std(y) > 1e-9) else None
        corr_after = corr_after if (corr_after is not None and np.isfinite(corr_after)) else None
        info["mae_before"] = mae_before
        info["mae_after"] = mae_after
        info["corr_after"] = corr_after

        # Apply only when mapping provides non-trivial MAE gain.
        if mae_after > mae_before * 0.99:
            info["reason"] = "mae_not_improved"
            return inet_v, info

        info["applied"] = True
        info["reason"] = "applied"
        return mapped_full.astype(float), info

    def evaluate_assim_quality(
        local_s: pd.Series,
        inet_s: pd.Series,
        hist_mask: pd.Series,
        corr_floor: float,
        *,
        min_points: int = 10,
        min_history_points: int = 24,
    ) -> dict[str, Any]:
        local_v = pd.to_numeric(local_s, errors="coerce").astype(float)
        inet_v = pd.to_numeric(inet_s, errors="coerce").astype(float)
        m, basis = choose_overlap_mask(
            local_v=local_v,
            inet_v=inet_v,
            hist_mask=hist_mask,
            min_points=min_points,
            min_history_points=min_history_points,
        )
        out_q: dict[str, Any] = {
            "basis": basis,
            "points": int(m.sum()),
            "corr_effective": None,
            "mae_effective": None,
            "nmae_effective": None,
            "corr_floor": float(corr_floor),
            "score_corr": 0.0,
            "score_error": 0.0,
            "reliability": 0.0,
            "passed": False,
        }
        if int(m.sum()) < int(max(6, min_points)):
            return out_q

        x = local_v[m].to_numpy(dtype=float)
        y = inet_v[m].to_numpy(dtype=float)
        if (np.std(x) > 1e-9) and (np.std(y) > 1e-9):
            corr = float(np.corrcoef(x, y)[0, 1])
            corr = corr if np.isfinite(corr) else None
        else:
            corr = None
        mae = float(np.mean(np.abs(x - y)))
        span = float(np.nanpercentile(x, 95.0) - np.nanpercentile(x, 5.0))
        span = max(span, 1e-6)
        nmae = float(mae / span)

        score_corr = 0.0
        if corr is not None:
            score_corr = float(np.clip((corr - float(corr_floor)) / max(1e-6, 1.0 - float(corr_floor)), 0.0, 1.0))
        score_error = float(np.clip(1.0 - 0.5 * nmae, 0.0, 1.0))
        reliability = float(score_corr * (0.65 + 0.35 * score_error))
        passed = bool(reliability >= 0.05)

        out_q.update(
            {
                "corr_effective": corr,
                "mae_effective": mae,
                "nmae_effective": nmae,
                "score_corr": score_corr,
                "score_error": score_error,
                "reliability": reliability,
                "passed": passed,
            }
        )
        return out_q

    for local_col, inet_col, w_attr, lo, hi, unit in configs:
        local = pd.to_numeric(out[local_col], errors="coerce").astype(float)
        out[f"{local_col}_local"] = local
        out[f"{local_col}_model"] = local.copy()
        out[f"{local_col}_assimilation_weight"] = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
        w_req = float(np.clip(float(getattr(args, w_attr)), 0.0, 1.0))
        vmeta: dict[str, Any] = {
            "blend_weight_requested": w_req,
            "blend_weight_applied": 0.0,
            "blend_weight_pre_decay": 0.0,
            "applied": False,
            "reason": f"internet_{local_col}_missing_or_weight_zero",
            "alignment": {"attempted": False, "applied": False, "reason": "not_attempted"},
        }

        if (inet_col in out.columns) and (w_req > 0):
            inet_raw = pd.to_numeric(out[inet_col], errors="coerce").astype(float)
            inet_h, align_info = harmonize_internet_series(local, inet_raw, ~fc)
            vmeta["alignment"] = align_info
            scale_ratio = align_info.get("scale_ratio")
            severe_scale_mismatch = (
                (scale_ratio is not None) and (np.isfinite(scale_ratio)) and ((scale_ratio < 0.50) or (scale_ratio > 2.00))
            )
            if severe_scale_mismatch and (not bool(align_info.get("applied", False))):
                # Keep raw internet signal for auditability, but mark harmonized channel invalid.
                out[f"{inet_col}_harmonized"] = np.nan
                out[f"{inet_col}_bias_corrected"] = np.nan
                out[f"{inet_col}_qm_corrected"] = np.nan
                out[local_col] = out[f"{local_col}_local"]
                vmeta["reason"] = "unit_mismatch_alignment_failed"
                meta["variables"][local_col] = vmeta
                continue

            inet_qm, qm_info = quantile_map_correct(
                local_s=local,
                inet_s=inet_h,
                hist_mask=~fc,
            )
            out[f"{inet_col}_qm_corrected"] = inet_qm.astype(float)
            vmeta["quantile_mapping"] = qm_info

            inet_adj, bias_info = seasonal_bias_correct(
                local_s=local,
                inet_s=inet_qm,
                ts_s=out["timestamp"],
                hist_mask=~fc,
            )
            out[f"{inet_col}_harmonized"] = inet_adj.astype(float)
            out[f"{inet_col}_bias_corrected"] = inet_adj.astype(float)
            vmeta["bias_correction"] = bias_info

            corr_floor = float(min_corr_floor.get(local_col, 0.1))
            qmeta = evaluate_assim_quality(
                local_s=local,
                inet_s=inet_adj,
                hist_mask=~fc,
                corr_floor=corr_floor,
            )
            vmeta["quality_gate"] = qmeta
            rel = float(qmeta.get("reliability", 0.0))
            rel_floor = float(np.clip(min_weight_fraction.get(local_col, 0.0), 0.0, 1.0))
            rel_eff = max(rel, rel_floor)
            w_pre_decay = float(np.clip(w_req * rel_eff, 0.0, 1.0))
            vmeta["weight_policy"] = {
                "reliability_raw": rel,
                "reliability_floor": rel_floor,
                "reliability_effective": rel_eff,
            }
            gate_pass = bool(qmeta.get("passed", False)) or (rel_floor >= 1.0)
            if (not gate_pass) or (w_pre_decay < 0.05):
                out[local_col] = out[f"{local_col}_local"]
                vmeta["reason"] = "low_reliability_weight"
                meta["variables"][local_col] = vmeta
                continue
            model = local.copy()
            m = fc & inet_adj.notna() & local.notna()
            if m.any():
                w_row = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
                w_row[m] = np.clip(w_pre_decay * lead_decay[m], 0.0, 1.0).astype(float)
                model[m] = (1.0 - w_row[m]) * local[m] + w_row[m] * inet_adj[m]
                model = np.clip(model, lo, hi)
                out[f"{local_col}_model"] = model.astype(float)
                out[local_col] = out[f"{local_col}_model"]
                out[f"{local_col}_assimilation_weight"] = w_row.astype(float)
                delta = (out[f"{local_col}_model"] - out[f"{local_col}_local"]).astype(float)
                delta_fc = delta[fc]
                vmeta.update(
                    {
                        "applied": True,
                        "blend_weight_pre_decay": w_pre_decay,
                        "blend_weight_applied": float(w_row[m].mean()),
                        "blend_weight_applied_p10": float(w_row[m].quantile(0.10)),
                        "blend_weight_applied_p90": float(w_row[m].quantile(0.90)),
                        "reason": "applied",
                        f"adjustment_mean_{unit}": float(delta_fc.mean()) if len(delta_fc) else 0.0,
                        f"adjustment_abs_mean_{unit}": float(np.mean(np.abs(delta_fc))) if len(delta_fc) else 0.0,
                        f"adjustment_p95_{unit}": float(delta_fc.abs().quantile(0.95)) if len(delta_fc) else 0.0,
                    }
                )
                any_applied = True
            else:
                out[local_col] = out[f"{local_col}_local"]
                vmeta["reason"] = "no_valid_forecast_overlap"
        else:
            if inet_col in out.columns:
                out[f"{inet_col}_harmonized"] = pd.to_numeric(out[inet_col], errors="coerce").astype(float)
                out[f"{inet_col}_bias_corrected"] = pd.to_numeric(out[inet_col], errors="coerce").astype(float)
                out[f"{inet_col}_qm_corrected"] = pd.to_numeric(out[inet_col], errors="coerce").astype(float)
            out[local_col] = out[f"{local_col}_local"]

        meta["variables"][local_col] = vmeta

    if any_applied:
        meta["enabled"] = True
        meta["reason"] = "applied"
    # Backward-compatible top-level temp fields.
    tmeta = meta["variables"].get("temp", {})
    meta["temp_blend_weight_requested"] = float(tmeta.get("blend_weight_requested", 0.0))
    meta["temp_blend_weight_applied"] = float(tmeta.get("blend_weight_applied", 0.0))
    meta["temp_applied"] = bool(tmeta.get("applied", False))
    meta["temp_adjustment_mean_c"] = tmeta.get("adjustment_mean_c")
    meta["temp_adjustment_abs_mean_c"] = tmeta.get("adjustment_abs_mean_c")
    meta["temp_adjustment_p95_c"] = tmeta.get("adjustment_p95_c")
    return out, meta


def apply_forecast_shortwave_assimilation(
    p10: pd.Series,
    p50: pd.Series,
    p90: pd.Series,
    physical_cap: np.ndarray,
    pv_temp_eff: np.ndarray,
    shortwave_internet_kwh: np.ndarray | None,
    is_forecast: pd.Series,
    blend_weight: float,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    w = float(np.clip(blend_weight, 0.0, 1.0))
    fc = is_forecast.astype(bool).to_numpy()
    out10 = p10.astype(float).to_numpy().copy()
    out50 = p50.astype(float).to_numpy().copy()
    out90 = p90.astype(float).to_numpy().copy()
    meta: dict[str, Any] = {
        "enabled": False,
        "blend_weight_requested": w,
        "blend_weight_applied": 0.0,
        "reason": "shortwave_missing_or_weight_zero",
    }
    if (shortwave_internet_kwh is None) or (w <= 0):
        return (
            pd.Series(out10, index=p10.index),
            pd.Series(out50, index=p50.index),
            pd.Series(out90, index=p90.index),
            meta,
        )

    sw = np.asarray(shortwave_internet_kwh, dtype=float)
    sw = np.clip(sw, 0.0, None)
    pv_eff = np.clip(np.asarray(pv_temp_eff, dtype=float), 0.72, 1.08)
    target = np.clip(sw * pv_eff, 0.0, np.asarray(physical_cap, dtype=float))

    m = fc & np.isfinite(target)
    if not np.any(m):
        return (
            pd.Series(out10, index=p10.index),
            pd.Series(out50, index=p50.index),
            pd.Series(out90, index=p90.index),
            meta,
        )

    old50 = out50.copy()
    old_width_lo = np.maximum(0.0, out50 - out10)
    old_width_hi = np.maximum(0.0, out90 - out50)
    out50[m] = (1.0 - w) * out50[m] + w * target[m]
    out50 = np.clip(out50, 0.0, np.asarray(physical_cap, dtype=float))
    out10 = np.maximum(0.0, out50 - old_width_lo)
    out90 = np.minimum(np.asarray(physical_cap, dtype=float), out50 + old_width_hi)
    out10 = np.minimum(out10, out50)
    out90 = np.maximum(out90, out50)

    delta = out50[m] - old50[m]
    meta.update(
        {
            "enabled": True,
            "blend_weight_applied": w,
            "reason": "applied",
            "adjustment_mean_kwh_m2_day": float(np.mean(delta)),
            "adjustment_abs_mean_kwh_m2_day": float(np.mean(np.abs(delta))),
            "adjustment_p95_kwh_m2_day": float(np.quantile(np.abs(delta), 0.95)),
        }
    )
    return (
        pd.Series(out10, index=p10.index),
        pd.Series(out50, index=p50.index),
        pd.Series(out90, index=p90.index),
        meta,
    )


def robust_norm_1d(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(s), 0.5, dtype=float), index=s.index)
    x = (s.astype(float) - lo) / (hi - lo)
    return pd.Series(np.clip(x.values, 0.0, 1.0), index=s.index)


def robust_norm_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        return np.full_like(x, 0.5, dtype=float)
    y = (x.astype(float) - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def classify_potential(index_vals: pd.Series) -> pd.Series:
    x = index_vals.astype(float)
    bins = [x < 30, (x >= 30) & (x < 50), (x >= 50) & (x < 70), x >= 70]
    labels = ["low", "medium", "high", "very_high"]
    return pd.Series(np.select(bins, labels, default="unknown"), index=x.index)


def calc_solar_geometry(ts: pd.Series, latitude_deg: float) -> pd.DataFrame:
    """Compute day length and extraterrestrial radiation from FAO-56 equations."""
    mid = pd.to_datetime(ts).dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=14)
    j = mid.dt.dayofyear.astype(float).values
    lat = math.radians(float(latitude_deg))
    gsc = 0.0820  # MJ m-2 min-1

    dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * j)
    delta = 0.409 * np.sin((2.0 * np.pi / 365.0) * j - 1.39)
    ws_arg = -np.tan(lat) * np.tan(delta)
    ws_arg = np.clip(ws_arg, -1.0, 1.0)
    ws = np.arccos(ws_arg)

    ra_mj = (24.0 * 60.0 / np.pi) * gsc * dr * (
        ws * np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.sin(ws)
    )
    daylen = 24.0 / np.pi * ws
    ra_kwh = ra_mj * 0.2777777778

    return pd.DataFrame(
        {
            "daylight_hours": daylen.astype(float),
            "extra_radiation_mj_m2_day": ra_mj.astype(float),
            "extra_radiation_kwh_m2_day": ra_kwh.astype(float),
        }
    )


def load_solar_observations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "solar_obs"])
    raw = read_table(path)
    ts_col = pick_column(raw, ["timestamp", "ds", "date", "datetime", "time", "tarih"])
    val_col = pick_column(raw, ["value", "yhat", "forecast", "solar", "solar_value"])
    var_col = pick_column(raw, ["variable", "metric", "name", "param"])
    if ts_col is None or val_col is None:
        return pd.DataFrame(columns=["timestamp", "solar_obs"])

    work = raw.copy()
    if var_col is not None:
        v = work[var_col].astype(str).str.strip().str.lower()
        work = work[v.isin({"solar", "gunes", "güneş", "radiation", "sun"})]
    if work.empty:
        return pd.DataFrame(columns=["timestamp", "solar_obs"])

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(work[ts_col], errors="coerce"),
            "solar_obs": pd.to_numeric(work[val_col], errors="coerce"),
        }
    ).dropna(subset=["timestamp", "solar_obs"])
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "solar_obs"])
    out = (
        out.set_index("timestamp")["solar_obs"]
        .resample("MS")
        .mean()
        .dropna()
        .rename("solar_obs")
        .reset_index()
    )
    return out


def fetch_nasa_power_monthly_solar(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout_seconds: int,
    cache_csv: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_csv = Path(cache_csv)
    meta: dict[str, Any] = {
        "provider": "nasa_power",
        "cache_csv": str(cache_csv),
        "used_cache": False,
        "download_url": None,
        "download_transport": None,
    }
    required_cols = {"timestamp", "solar_obs"}

    if cache_csv.exists() and (not refresh):
        try:
            cached = pd.read_csv(cache_csv, parse_dates=["timestamp"])
            if required_cols.issubset(set(cached.columns)):
                cached = cached.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)
                meta["used_cache"] = True
                meta["rows"] = int(len(cached))
                return cached[["timestamp", "solar_obs"]].copy(), meta
        except Exception:
            pass

    start_year = int(pd.to_datetime(start_date).year)
    requested_end_year = int(pd.to_datetime(end_date).year)
    current_year = int(pd.Timestamp.utcnow().year)
    end_year = min(requested_end_year, current_year)

    payload: dict[str, Any] | None = None
    last_err: Exception | None = None
    used_end_year: int | None = None
    for y_end in range(end_year, max(start_year, end_year - 5) - 1, -1):
        url = (
            "https://power.larc.nasa.gov/api/temporal/monthly/point?"
            + urlencode(
                {
                    "parameters": "ALLSKY_SFC_SW_DWN",
                    "community": "RE",
                    "longitude": f"{float(longitude):.4f}",
                    "latitude": f"{float(latitude):.4f}",
                    "start": str(start_year),
                    "end": str(y_end),
                    "format": "JSON",
                }
            )
        )
        meta["download_url"] = url
        try:
            with urlopen(url, timeout=max(5, int(timeout_seconds))) as resp:
                payload_try = json.loads(resp.read().decode("utf-8"))
            param_try = ((payload_try.get("properties") or {}).get("parameter") or {}).get("ALLSKY_SFC_SW_DWN")
            if isinstance(param_try, dict) and param_try:
                payload = payload_try
                meta["download_transport"] = "urllib"
                used_end_year = int(y_end)
                break
            last_err = RuntimeError(f"missing_parameter_for_end_year_{y_end}")
            continue
        except Exception as exc:
            last_err = exc
            try:
                proc = subprocess.run(
                    ["curl", "-sS", url],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=max(10, int(timeout_seconds) + 5),
                )
                payload_try = json.loads(proc.stdout)
                param_try = ((payload_try.get("properties") or {}).get("parameter") or {}).get("ALLSKY_SFC_SW_DWN")
                if isinstance(param_try, dict) and param_try:
                    payload = payload_try
                    meta["download_transport"] = "curl_fallback"
                    used_end_year = int(y_end)
                    break
                last_err = RuntimeError(f"missing_parameter_for_end_year_{y_end}")
                continue
            except Exception as exc2:
                last_err = exc2
                continue

    if payload is None:
        raise RuntimeError(f"NASA POWER fetch failed: {last_err}")

    param = ((payload.get("properties") or {}).get("parameter") or {}).get("ALLSKY_SFC_SW_DWN")
    if not isinstance(param, dict) or (not param):
        raise RuntimeError("NASA POWER response missing ALLSKY_SFC_SW_DWN.")

    rows: list[dict[str, Any]] = []
    for k, v in param.items():
        try:
            ts = pd.to_datetime(str(k), format="%Y%m", errors="coerce")
        except Exception:
            ts = pd.NaT
        val = pd.to_numeric(v, errors="coerce")
        if pd.isna(ts) or pd.isna(val):
            continue
        fval = float(val)
        if fval < 0:
            continue
        rows.append({"timestamp": ts, "solar_obs": fval})

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("NASA POWER returned empty monthly series.")
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_csv, index=False)
    meta["rows"] = int(len(out))
    meta["used_end_year"] = int(used_end_year) if used_end_year is not None else None
    return out, meta


def build_solar_observation_set(merged: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    local_obs = load_solar_observations(args.solar_observations)
    mode = str(args.solar_reference_mode).strip().lower()
    meta: dict[str, Any] = {
        "mode": mode,
        "local_rows": int(len(local_obs)),
        "selected": "local",
        "reference": {"enabled": False},
    }
    if mode == "none":
        return local_obs, meta

    need_ref = bool(mode == "nasa_power" or (mode == "auto" and len(local_obs) < int(args.min_calibration_points)))
    if not need_ref:
        return local_obs, meta

    ts = pd.to_datetime(merged["timestamp"], errors="coerce")
    start_date = str(ts.min().date())
    end_date = str(ts.max().date())
    refresh = bool(to_bool_text(args.refresh_solar_reference))
    try:
        ref_obs, ref_meta = fetch_nasa_power_monthly_solar(
            latitude=float(args.latitude),
            longitude=float(args.longitude),
            start_date=start_date,
            end_date=end_date,
            timeout_seconds=int(args.internet_timeout_seconds),
            cache_csv=Path(args.nasa_power_cache_csv),
            refresh=refresh,
        )
        ref_obs = ref_obs.copy()
        ref_obs["source"] = "nasa_power"
        if len(local_obs):
            loc = local_obs.copy()
            loc["source"] = "local"
            all_obs = pd.concat([ref_obs, loc], ignore_index=True)
            all_obs["rank"] = all_obs["source"].map({"nasa_power": 0, "local": 1}).fillna(0).astype(int)
            all_obs = (
                all_obs.sort_values(["timestamp", "rank"])
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            solar_obs = all_obs[["timestamp", "solar_obs"]].copy()
            selected = "merged_local_plus_nasa_power"
        else:
            solar_obs = ref_obs[["timestamp", "solar_obs"]].copy()
            selected = "nasa_power_only"

        meta["selected"] = selected
        meta["rows_final"] = int(len(solar_obs))
        meta["reference"] = {
            "enabled": True,
            "provider": "nasa_power",
            "rows": int(len(ref_obs)),
            "refresh": refresh,
            "fetch": ref_meta,
        }
        return solar_obs, meta
    except Exception as exc:
        meta["reference"] = {"enabled": False, "error": str(exc)}
        if mode == "nasa_power":
            raise SystemExit(f"NASA POWER solar reference fetch failed: {exc}")
        return local_obs, meta


def calibrate_series(
    pred: pd.Series,
    obs: pd.Series,
    low_point_shrink: float = 0.35,
    min_points: int = 6,
) -> CalibrationInfo:
    x = pred.astype(float).values
    y = obs.astype(float).values
    n = len(x)
    if n == 0:
        return CalibrationInfo(False, 0, 1.0, 0.0, float("nan"), float("nan"), "none")
    if n < int(max(1, min_points)):
        return CalibrationInfo(False, n, 1.0, 0.0, float("nan"), float("nan"), f"insufficient_points_{n}")

    if n < 3:
        # Low-data fallback: only bias correction.
        slope = 1.0
        raw_intercept = float(np.mean(y - x))
        shrink = float(np.clip(low_point_shrink, 0.0, 1.0))
        intercept = raw_intercept * shrink
        yh = x + intercept
        mae = float(np.mean(np.abs(y - yh)))
        return CalibrationInfo(True, n, slope, intercept, mae, float("nan"), f"bias_shrunk_{n}pt")

    slope, intercept = np.polyfit(x, y, 1)
    yh = slope * x + intercept
    mae = float(np.mean(np.abs(y - yh)))
    den = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((y - yh) ** 2) / den) if den > 0 else float("nan")
    return CalibrationInfo(True, n, float(slope), float(intercept), mae, r2, "linear")


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    month = ts.dt.month.astype(float)
    m_sin = np.sin(2.0 * np.pi * month / 12.0)
    m_cos = np.cos(2.0 * np.pi * month / 12.0)
    year_frac = ts.dt.year.astype(float) + (ts.dt.month.astype(float) - 1.0) / 12.0
    trend = (year_frac - float(year_frac.min())).astype(float)
    feats = pd.DataFrame(
        {
            "temp": df["temp"].astype(float),
            "humidity": df["humidity"].astype(float),
            "precip": df["precip"].astype(float),
            "pressure": df["pressure"].astype(float),
            "daylight_hours": df["daylight_hours"].astype(float),
            "extra_radiation_kwh_m2_day": df["extra_radiation_kwh_m2_day"].astype(float),
            "heuristic_kwh": df["heuristic_kwh_m2_day"].astype(float),
            "month_sin": m_sin.astype(float),
            "month_cos": m_cos.astype(float),
            "year_trend": trend,
            "temp_x_humidity": (df["temp"] * df["humidity"]).astype(float),
            "temp_x_precip": (df["temp"] * df["precip"]).astype(float),
            "humidity_x_precip": (df["humidity"] * df["precip"]).astype(float),
        }
    )
    if "cloud_cover_internet_pct" in df.columns:
        feats["cloud_cover_internet_pct"] = pd.to_numeric(df["cloud_cover_internet_pct"], errors="coerce").astype(float)
    if "wind_speed_internet_kmh" in df.columns:
        feats["wind_speed_internet_kmh"] = pd.to_numeric(df["wind_speed_internet_kmh"], errors="coerce").astype(float)
    if "shortwave_internet_kwh_m2_day" in df.columns:
        feats["shortwave_internet_kwh_m2_day"] = pd.to_numeric(df["shortwave_internet_kwh_m2_day"], errors="coerce").astype(float)
    if "temp_internet_c" in df.columns:
        feats["temp_internet_c"] = pd.to_numeric(df["temp_internet_c"], errors="coerce").astype(float)
    if "humidity_internet_pct" in df.columns:
        feats["humidity_internet_pct"] = pd.to_numeric(df["humidity_internet_pct"], errors="coerce").astype(float)
    if "precip_internet_mm_day" in df.columns:
        feats["precip_internet_mm_day"] = pd.to_numeric(df["precip_internet_mm_day"], errors="coerce").astype(float)
    if "pressure_internet" in df.columns:
        feats["pressure_internet"] = pd.to_numeric(df["pressure_internet"], errors="coerce").astype(float)

    # Time-lag and rolling features improve monthly sequence learning.
    sort_idx = ts.sort_values().index
    lag_cfg = [
        "temp",
        "humidity",
        "precip",
        "pressure",
        "heuristic_kwh",
        "extra_radiation_kwh_m2_day",
    ]
    for col in lag_cfg:
        if col not in feats.columns:
            continue
        s = feats.loc[sort_idx, col].astype(float)
        feats.loc[sort_idx, f"{col}_lag1"] = s.shift(1).values
        feats.loc[sort_idx, f"{col}_lag12"] = s.shift(12).values
        feats.loc[sort_idx, f"{col}_ma3"] = s.shift(1).rolling(3, min_periods=1).mean().values
        feats.loc[sort_idx, f"{col}_ma12"] = s.shift(1).rolling(12, min_periods=3).mean().values

    inet_lag_cfg = [
        "temp_internet_c",
        "humidity_internet_pct",
        "precip_internet_mm_day",
        "shortwave_internet_kwh_m2_day",
        "cloud_cover_internet_pct",
    ]
    for col in inet_lag_cfg:
        if col not in feats.columns:
            continue
        s = feats.loc[sort_idx, col].astype(float)
        feats.loc[sort_idx, f"{col}_lag1"] = s.shift(1).values
        feats.loc[sort_idx, f"{col}_ma3"] = s.shift(1).rolling(3, min_periods=1).mean().values

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.interpolate(method="linear", limit_direction="both")
    feats = feats.ffill().bfill()
    feats = feats.fillna(0.0)
    return feats


def run_ml_correction(
    full_df: pd.DataFrame,
    solar_obs: pd.DataFrame,
    min_months: int,
    holdout: int,
) -> tuple[pd.Series | None, pd.Series | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": False,
        "reason": "",
        "train_rows": 0,
        "holdout_rows": 0,
        "feature_count": 0,
        "weighting_method": "",
        "cv_folds": 0,
        "cv_window_size": 0,
        "cv_mae": {},
        "models": [],
        "weights": {},
        "valid_mae": None,
        "valid_rmse": None,
        "valid_mae_ensemble": None,
        "valid_rmse_ensemble": None,
        "valid_mae_baseline": None,
        "valid_rmse_baseline": None,
        "post_blend": {
            "enabled": False,
            "applied": False,
            "reason": "not_attempted",
            "w_ml_raw": None,
            "w_ml_applied": None,
            "mae_mixed": None,
            "rmse_mixed": None,
        },
        "conformal": {
            "enabled": False,
            "q10_residual": None,
            "q90_residual": None,
            "coverage_empirical_10_90": None,
            "interval_width_mean": None,
        },
    }
    if not SKLEARN_OK:
        meta["reason"] = "sklearn_not_available"
        return None, None, meta

    joined = full_df.merge(solar_obs, on="timestamp", how="inner")
    n = len(joined)
    meta["train_rows"] = int(n)
    if n < max(int(min_months), 8):
        meta["reason"] = f"insufficient_target_months:{n}"
        return None, None, meta

    feats = build_ml_features(joined)
    meta["feature_count"] = int(feats.shape[1])
    y = joined["solar_obs"].astype(float).values

    h = int(max(6, min(int(holdout), n // 3)))
    split = n - h
    if split < 6:
        meta["reason"] = "too_short_after_holdout"
        return None, None, meta
    meta["holdout_rows"] = int(h)

    x_tr = feats.iloc[:split].values
    y_tr = y[:split]
    x_va = feats.iloc[split:].values
    y_va = y[split:]

    model_builders: dict[str, Any] = {
        "ridge": lambda: Ridge(alpha=1.0, random_state=42),
        "gbr_d3": lambda: GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
            loss="squared_error",
        ),
        "gbr_d2": lambda: GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=2,
            random_state=42,
            loss="squared_error",
        ),
        "rf_d6": lambda: RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_d8": lambda: RandomForestRegressor(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_deep": lambda: RandomForestRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    model_names = list(model_builders.keys())

    def make_model(name: str) -> Any:
        if name not in model_builders:
            raise ValueError(f"unknown_model:{name}")
        return model_builders[name]()

    # Time-series CV on development window for robust ensemble weighting.
    cv_mae_lists: dict[str, list[float]] = {k: [] for k in model_names}
    cv_min_train = int(max(24, min(split - 6, 60)))
    cv_window = int(max(6, min(24, split // 6)))
    cv_starts = list(range(cv_min_train, split, cv_window))
    for vs in cv_starts:
        ve = int(min(split, vs + cv_window))
        if (ve - vs) < 3 or vs < 12:
            continue
        x_cv_tr = feats.iloc[:vs].values
        y_cv_tr = y[:vs]
        x_cv_va = feats.iloc[vs:ve].values
        y_cv_va = y[vs:ve]
        for name in model_names:
            mdl = make_model(name)
            mdl.fit(x_cv_tr, y_cv_tr)
            pred = np.asarray(mdl.predict(x_cv_va), dtype=float)
            cv_mae_lists[name].append(float(np.mean(np.abs(y_cv_va - pred))))

    cv_folds = int(min(len(v) for v in cv_mae_lists.values())) if cv_mae_lists else 0
    meta["cv_folds"] = int(cv_folds)
    meta["cv_window_size"] = int(cv_window)
    meta["cv_mae"] = {k: (float(np.mean(v)) if len(v) else None) for k, v in cv_mae_lists.items()}

    if cv_folds >= 2:
        base_err = {k: float(np.median(v)) for k, v in cv_mae_lists.items()}
        meta["weighting_method"] = "timeseries_cv_inverse_mae"
    else:
        # Fallback: single holdout MAE for weight estimation.
        base_err = {}
        for name in model_names:
            mdl = make_model(name)
            mdl.fit(x_tr, y_tr)
            pred = np.asarray(mdl.predict(x_va), dtype=float)
            base_err[name] = float(np.mean(np.abs(y_va - pred)))
        meta["weighting_method"] = "single_holdout_inverse_mae"

    # Holdout evaluation with development-fitted models (no leakage).
    va_pred: dict[str, np.ndarray] = {}
    holdout_metrics: dict[str, dict[str, float]] = {}
    for name in model_names:
        mdl = make_model(name)
        mdl.fit(x_tr, y_tr)
        pred = np.asarray(mdl.predict(x_va), dtype=float)
        va_pred[name] = pred
        holdout_metrics[name] = {
            "mae": float(np.mean(np.abs(y_va - pred))),
            "rmse": float(np.sqrt(np.mean((y_va - pred) ** 2))),
        }

    rank_by_mae = sorted(model_names, key=lambda k: holdout_metrics[k]["mae"])
    best_single_name = rank_by_mae[0]
    best_single_mae = float(holdout_metrics[best_single_name]["mae"])
    best_single_rmse = float(holdout_metrics[best_single_name]["rmse"])

    selection_lambda_rmse = 0.45
    strategy_rows: list[dict[str, Any]] = []
    strategy_rows.append(
        {
            "strategy": "single_best",
            "members": [best_single_name],
            "weights": {best_single_name: 1.0},
            "mae": best_single_mae,
            "rmse": best_single_rmse,
            "score": float(best_single_mae + selection_lambda_rmse * best_single_rmse),
        }
    )

    max_members = int(min(5, len(rank_by_mae)))
    for k in range(2, max_members + 1):
        members = rank_by_mae[:k]
        inv = np.array([1.0 / max(float(base_err[m]), 1e-6) for m in members], dtype=float)
        den = float(np.sum(inv))
        if (not np.isfinite(den)) or den <= 0:
            continue
        w_arr = inv / den
        w = {m: float(wi) for m, wi in zip(members, w_arr)}
        pred = np.zeros_like(y_va, dtype=float)
        for m in members:
            pred += w[m] * va_pred[m]
        mae = float(np.mean(np.abs(y_va - pred)))
        rmse = float(np.sqrt(np.mean((y_va - pred) ** 2)))
        strategy_rows.append(
            {
                "strategy": f"ensemble_top{k}",
                "members": members,
                "weights": w,
                "mae": mae,
                "rmse": rmse,
                "score": float(mae + selection_lambda_rmse * rmse),
            }
        )

    # Positive-weight holdout stacking blend (NNLS-like) for bias-variance balance.
    if len(model_names) >= 2 and len(y_va) >= 12:
        try:
            v_mat = np.column_stack([va_pred[m] for m in model_names]).astype(float)
            stacker = LinearRegression(fit_intercept=False, positive=True)
            stacker.fit(v_mat, y_va)
            w_raw = np.clip(np.asarray(stacker.coef_, dtype=float), 0.0, None)
            sw = float(np.sum(w_raw))
            if np.isfinite(sw) and sw > 0:
                w_raw = w_raw / sw
                members: list[str] = []
                w: dict[str, float] = {}
                for m, ww in zip(model_names, w_raw):
                    if float(ww) <= 1e-4:
                        continue
                    members.append(str(m))
                    w[str(m)] = float(ww)
                if members:
                    pred = np.zeros_like(y_va, dtype=float)
                    for m in members:
                        pred += w[m] * va_pred[m]
                    mae = float(np.mean(np.abs(y_va - pred)))
                    rmse = float(np.sqrt(np.mean((y_va - pred) ** 2)))
                    strategy_rows.append(
                        {
                            "strategy": "nnls_holdout_blend",
                            "members": members,
                            "weights": w,
                            "mae": mae,
                            "rmse": rmse,
                            "score": float(mae + selection_lambda_rmse * rmse),
                        }
                    )
        except Exception:
            pass

    strategy_rows.sort(key=lambda r: (float(r["score"]), float(r["mae"])))
    selected = strategy_rows[0]
    selection_reason = "score_minimum"
    if str(selected.get("strategy", "")).startswith("ensemble_"):
        mae_gain = (best_single_mae - float(selected["mae"])) / max(best_single_mae, 1e-9)
        rmse_ratio = float(selected["rmse"]) / max(best_single_rmse, 1e-9)
        if ((mae_gain < 0.004) and (rmse_ratio > 1.01)) or (rmse_ratio > 1.08):
            selected = next(r for r in strategy_rows if r["strategy"] == "single_best")
            selection_reason = "fallback_single_for_rmse_or_small_mae_gain"

    selected_members = [str(m) for m in selected["members"]]
    selected_weights = {str(k): float(v) for k, v in selected["weights"].items()}
    valid_mae_ens = float(selected["mae"])
    valid_rmse_ens = float(selected["rmse"])
    va_blend = np.zeros_like(y_va, dtype=float)
    for m in selected_members:
        va_blend += selected_weights[m] * va_pred[m]

    x_full = build_ml_features(full_df).values
    y_all_ens = np.zeros(len(full_df), dtype=float)
    member_preds: list[np.ndarray] = []
    # Final training on all available observed-target rows.
    x_fit_all = feats.values
    y_fit_all = y
    for name in selected_members:
        mdl = make_model(name)
        mdl.fit(x_fit_all, y_fit_all)
        pred_f = np.asarray(mdl.predict(x_full), dtype=float)
        member_preds.append(pred_f)
        y_all_ens += selected_weights[name] * pred_f

    if len(member_preds) >= 2:
        spread_ens = np.std(np.vstack(member_preds), axis=0)
    else:
        resid_single = y_va - va_blend
        resid_sd = float(np.std(resid_single, ddof=1)) if len(resid_single) > 2 else float(np.std(resid_single))
        spread_ens = np.full(len(full_df), max(0.0, 0.35 * resid_sd), dtype=float)
    y_all = y_all_ens.copy()
    va_final = va_blend.copy()
    valid_mae = valid_mae_ens
    valid_rmse = valid_rmse_ens
    spread = spread_ens.copy()

    meta["selection_metric"] = "mae_plus_0.45_rmse"
    meta["selection_reason"] = selection_reason
    meta["candidate_holdout"] = holdout_metrics
    meta["strategy_candidates"] = [
        {
            "strategy": str(r["strategy"]),
            "members": [str(m) for m in r["members"]],
            "weights": {str(k): float(v) for k, v in r["weights"].items()},
            "mae": float(r["mae"]),
            "rmse": float(r["rmse"]),
            "score": float(r["score"]),
        }
        for r in strategy_rows
    ]
    meta["selected_strategy"] = str(selected["strategy"])
    meta["selected_models"] = selected_members
    meta["selected_weights"] = selected_weights

    if "heuristic_kwh_m2_day" in full_df.columns and "heuristic_kwh_m2_day" in joined.columns:
        base_all = pd.to_numeric(full_df["heuristic_kwh_m2_day"], errors="coerce").astype(float).to_numpy()
        base_va = pd.to_numeric(joined["heuristic_kwh_m2_day"], errors="coerce").astype(float).to_numpy()[split:]
        m_base = np.isfinite(base_va)
        if int(np.sum(m_base)) >= max(6, int(0.6 * len(base_va))):
            mae_base = float(np.mean(np.abs(y_va[m_base] - base_va[m_base])))
            rmse_base = float(np.sqrt(np.mean((y_va[m_base] - base_va[m_base]) ** 2)))
            meta["valid_mae_baseline"] = mae_base
            meta["valid_rmse_baseline"] = rmse_base
            w_ml_raw = float(mae_base / max(1e-9, mae_base + valid_mae_ens))
            w_ml = float(np.clip(w_ml_raw, 0.35, 0.95))
            base_va_mix = np.where(np.isfinite(base_va), base_va, va_blend)
            va_mix = w_ml * va_blend + (1.0 - w_ml) * base_va_mix
            mae_mix = float(np.mean(np.abs(y_va - va_mix)))
            rmse_mix = float(np.sqrt(np.mean((y_va - va_mix) ** 2)))
            post_meta = {
                "enabled": True,
                "applied": False,
                "reason": "no_gain",
                "w_ml_raw": w_ml_raw,
                "w_ml_applied": None,
                "mae_mixed": mae_mix,
                "rmse_mixed": rmse_mix,
            }
            # Apply post-blend when it is not materially worse than ensemble holdout error.
            if mae_mix <= valid_mae_ens * 1.01:
                base_all_mix = np.where(np.isfinite(base_all), base_all, y_all_ens)
                y_all = w_ml * y_all_ens + (1.0 - w_ml) * base_all_mix
                y_all = np.clip(y_all, 0.0, None)
                va_final = va_mix
                valid_mae = mae_mix
                valid_rmse = rmse_mix
                spread_base = np.abs(y_all_ens - base_all_mix)
                spread = np.sqrt(np.maximum(0.0, (w_ml * spread_ens) ** 2 + ((1.0 - w_ml) * spread_base) ** 2))
                post_meta["applied"] = True
                post_meta["reason"] = "applied"
                post_meta["w_ml_applied"] = w_ml
            meta["post_blend"] = post_meta

    resid = (y_va - va_final).astype(float)
    resid_std = float(np.std(resid, ddof=1)) if len(y_va) > 2 else float(np.std(resid))
    total_unc = np.sqrt(np.maximum(0.0, spread**2 + resid_std**2))

    if len(resid) >= 8:
        q10 = float(np.quantile(resid, 0.10))
        q90 = float(np.quantile(resid, 0.90))
        in_int = ((resid >= q10) & (resid <= q90)).astype(float)
        cov = float(np.mean(in_int))
        meta["conformal"] = {
            "enabled": True,
            "q10_residual": q10,
            "q90_residual": q90,
            "coverage_empirical_10_90": cov,
            "interval_width_mean": float(np.mean(q90 - q10)),
        }

    meta["enabled"] = True
    meta["models"] = list(model_names)
    meta["weights"] = selected_weights
    meta["valid_mae_ensemble"] = valid_mae_ens
    meta["valid_rmse_ensemble"] = valid_rmse_ens
    meta["valid_mae"] = valid_mae
    meta["valid_rmse"] = valid_rmse
    return pd.Series(y_all, index=full_df.index), pd.Series(total_unc, index=full_df.index), meta


def resolve_gamma_pdc(module_type: str, gamma_user: float) -> float:
    if np.isfinite(gamma_user):
        return float(gamma_user)
    key = str(module_type).strip().lower()
    return float(PV_GAMMA_MAP.get(key, PV_GAMMA_MAP["standard"]))


def erbs_diffuse_fraction(kt: np.ndarray) -> np.ndarray:
    """Erbs et al. (1982) daily diffuse-fraction correlation."""
    x = np.clip(np.asarray(kt, dtype=float), 0.0, 1.2)
    kd = np.empty_like(x, dtype=float)

    m1 = x <= 0.22
    m2 = (x > 0.22) & (x <= 0.80)
    m3 = x > 0.80

    kd[m1] = 1.0 - 0.09 * x[m1]
    kd[m2] = (
        0.9511
        - 0.1604 * x[m2]
        + 4.3880 * x[m2] ** 2
        - 16.638 * x[m2] ** 3
        + 12.336 * x[m2] ** 4
    )
    kd[m3] = 0.165
    return np.clip(kd, 0.0, 1.0)


def compute_heuristic_components(
    temp: np.ndarray,
    humidity: np.ndarray,
    precip: np.ndarray,
    pressure: np.ndarray,
    extra_radiation_kwh: np.ndarray,
    temp_opt_c: float,
    temp_penalty_per_c: float,
    elevation_m: float,
    gamma_pdc: float,
    norm_bounds: dict[str, tuple[float, float]],
    cloud_cover_pct: np.ndarray | None = None,
    wind_speed_kmh: np.ndarray | None = None,
    shortwave_kwh: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    temp_score = np.exp(-((temp - temp_opt_c) / 14.0) ** 2)
    humidity_n = robust_norm_array(humidity, *norm_bounds["humidity"])
    precip_n = robust_norm_array(precip, *norm_bounds["precip"])
    pressure_n = robust_norm_array(pressure, *norm_bounds["pressure"])

    humidity_penalty = humidity_n
    precip_penalty = np.sqrt(np.clip(precip_n, 0.0, 1.0))
    pressure_bonus = 0.90 + 0.20 * pressure_n
    if cloud_cover_pct is None:
        cloud_cover_n = np.zeros_like(temp_score, dtype=float)
    else:
        cloud_cover_n = np.clip(np.asarray(cloud_cover_pct, dtype=float) / 100.0, 0.0, 1.0)

    # FAO-56 clear-sky upper envelope: Rso = (0.75 + 2e-5*z) * Ra
    rso_coeff = float(np.clip(0.75 + 2.0e-5 * float(elevation_m), 0.70, 0.90))
    rso_kwh = np.clip(extra_radiation_kwh * rso_coeff, 0.0, None)

    # Weather-driven clearness index proxy (kt = Rs/Ra).
    # Bounded in a realistic physical range for monthly means.
    logit = (
        -0.45
        + 0.95 * pressure_n
        + 0.45 * temp_score
        - 1.15 * humidity_penalty
        - 0.95 * precip_penalty
        - 1.10 * cloud_cover_n
    )
    sig = 1.0 / (1.0 + np.exp(-logit))
    kt_est = np.clip(0.15 + 0.70 * sig, 0.08, 0.82)

    # Global horizontal radiation Rs, constrained by Rso.
    rs_kwh = np.clip(kt_est * extra_radiation_kwh, 0.0, None)
    if shortwave_kwh is not None:
        sw = np.asarray(shortwave_kwh, dtype=float)
        sw = np.clip(sw, 0.0, None)
        rs_proxy = np.minimum(sw, rso_kwh)
        rs_kwh = np.where(np.isfinite(rs_proxy), 0.75 * rs_kwh + 0.25 * rs_proxy, rs_kwh)
    rs_kwh = np.minimum(rs_kwh, rso_kwh)
    kt_eff = np.divide(rs_kwh, np.maximum(extra_radiation_kwh, 1e-9))

    # Erbs diffuse decomposition for interpretability.
    kd = erbs_diffuse_fraction(kt_eff)
    diffuse_kwh = np.clip(kd * rs_kwh, 0.0, None)
    beam_kwh = np.clip(rs_kwh - diffuse_kwh, 0.0, None)

    cloud_loss = np.clip(1.0 - np.divide(rs_kwh, np.maximum(rso_kwh, 1e-9)), 0.0, 1.0)

    # PVWatts-style temperature derate (gamma typically negative).
    # Use ambient temperature proxy in absence of module temperature.
    pv_temp_eff = 1.0 + float(gamma_pdc) * (temp - 25.0)
    pv_temp_eff = np.clip(pv_temp_eff, 0.75, 1.08)
    if wind_speed_kmh is None:
        wind_cooling = np.ones_like(pv_temp_eff, dtype=float)
    else:
        wnd = np.clip(np.asarray(wind_speed_kmh, dtype=float), 0.0, None)
        wnd_n = np.clip(wnd / 25.0, 0.0, 1.0)
        wind_cooling = np.clip(1.0 + 0.04 * wnd_n, 0.96, 1.08)
    # Retain legacy knob as optional hot-condition safeguard.
    pv_temp_eff *= np.clip(1.0 - float(temp_penalty_per_c) * np.maximum(0.0, temp - temp_opt_c), 0.85, 1.05)
    pv_temp_eff *= wind_cooling
    pv_temp_eff = np.clip(pv_temp_eff, 0.72, 1.08)

    heuristic_kwh = np.clip(rs_kwh * pv_temp_eff, 0.0, None)

    return {
        "temp_score": temp_score,
        "humidity_penalty": humidity_penalty,
        "precip_penalty": precip_penalty,
        "pressure_bonus": pressure_bonus,
        "cloud_cover_penalty": cloud_cover_n,
        "clearness_index": kt_eff,
        "diffuse_fraction": kd,
        "clear_sky_kwh": rso_kwh,
        "cloud_loss": cloud_loss,
        "after_cloud_kwh": rs_kwh,
        "global_horizontal_kwh": rs_kwh,
        "diffuse_kwh": diffuse_kwh,
        "beam_kwh": beam_kwh,
        "wind_cooling_bonus": wind_cooling,
        "pv_temp_eff": pv_temp_eff,
        "heuristic_kwh": heuristic_kwh,
    }


def derive_sigma_series(
    value: pd.Series,
    low: pd.Series,
    high: pd.Series,
    variable: str,
    history_std: float,
    z: float,
) -> pd.Series:
    width_sigma = (high.astype(float) - low.astype(float)) / max(2.0 * float(z), 1e-6)
    valid_width = np.isfinite(width_sigma) & (width_sigma > 0)
    fallback = max(float(SIGMA_MIN[variable]), float(SIGMA_FRAC_STD[variable]) * max(float(history_std), 0.0))
    sigma = pd.Series(np.where(valid_width, width_sigma, fallback), index=value.index, dtype=float)
    sigma = sigma.clip(lower=float(SIGMA_MIN[variable]))
    return sigma


def run_monte_carlo(
    merged: pd.DataFrame,
    norm_bounds: dict[str, tuple[float, float]],
    calib: CalibrationInfo,
    sigma_map: dict[str, pd.Series],
    mc_samples: int,
    random_seed: int,
    temp_opt_c: float,
    temp_penalty_per_c: float,
    elevation_m: float,
    gamma_pdc: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(merged)
    m = int(max(50, mc_samples))
    rng = np.random.default_rng(int(random_seed))

    temp_mu = merged["temp"].astype(float).to_numpy()[:, None]
    hum_mu = merged["humidity"].astype(float).to_numpy()[:, None]
    prc_mu = merged["precip"].astype(float).to_numpy()[:, None]
    prs_mu = merged["pressure"].astype(float).to_numpy()[:, None]

    temp_sd = sigma_map["temp"].astype(float).to_numpy()[:, None]
    hum_sd = sigma_map["humidity"].astype(float).to_numpy()[:, None]
    prc_sd = sigma_map["precip"].astype(float).to_numpy()[:, None]
    prs_sd = sigma_map["pressure"].astype(float).to_numpy()[:, None]

    temp_sim = rng.normal(temp_mu, temp_sd, size=(n, m))
    hum_sim = rng.normal(hum_mu, hum_sd, size=(n, m))
    prc_sim = rng.normal(prc_mu, prc_sd, size=(n, m))
    prs_sim = rng.normal(prs_mu, prs_sd, size=(n, m))

    temp_sim = np.clip(temp_sim, -50.0, 65.0)
    hum_sim = np.clip(hum_sim, 0.0, 100.0)
    prc_sim = np.clip(prc_sim, 0.0, None)
    prs_sim = np.clip(prs_sim, 0.0, None)

    extra = merged["extra_radiation_kwh_m2_day"].astype(float).to_numpy()[:, None]
    cloud_extra = (
        merged["cloud_cover_internet_pct"].astype(float).to_numpy()[:, None]
        if "cloud_cover_internet_pct" in merged.columns
        else None
    )
    wind_extra = (
        merged["wind_speed_internet_kmh"].astype(float).to_numpy()[:, None]
        if "wind_speed_internet_kmh" in merged.columns
        else None
    )
    shortwave_extra = (
        merged["shortwave_internet_kwh_m2_day"].astype(float).to_numpy()[:, None]
        if "shortwave_internet_kwh_m2_day" in merged.columns
        else None
    )
    comp = compute_heuristic_components(
        temp=temp_sim,
        humidity=hum_sim,
        precip=prc_sim,
        pressure=prs_sim,
        extra_radiation_kwh=extra,
        temp_opt_c=float(temp_opt_c),
        temp_penalty_per_c=float(temp_penalty_per_c),
        elevation_m=float(elevation_m),
        gamma_pdc=float(gamma_pdc),
        norm_bounds=norm_bounds,
        cloud_cover_pct=cloud_extra,
        wind_speed_kmh=wind_extra,
        shortwave_kwh=shortwave_extra,
    )

    sim_kwh = np.clip(comp["heuristic_kwh"], 0.0, None)
    if calib.enabled:
        sim_kwh = np.clip(sim_kwh * float(calib.slope) + float(calib.intercept), 0.0, None)

    p10 = pd.Series(np.quantile(sim_kwh, 0.10, axis=1), index=merged.index)
    p50 = pd.Series(np.quantile(sim_kwh, 0.50, axis=1), index=merged.index)
    p90 = pd.Series(np.quantile(sim_kwh, 0.90, axis=1), index=merged.index)
    return p10, p50, p90


def smooth_forecast_only(values: pd.Series, is_forecast: pd.Series, alpha: float) -> pd.Series:
    a = float(alpha)
    if a <= 0:
        return values.astype(float).copy()
    a = min(a, 1.0)

    out = values.astype(float).to_numpy().copy()
    fc = is_forecast.astype(bool).to_numpy()
    if not fc.any():
        return pd.Series(out, index=values.index)

    start = int(np.argmax(fc))
    prev = float(out[start - 1]) if start > 0 else float(out[start])
    for i in range(start, len(out)):
        if not fc[i]:
            prev = float(out[i])
            continue
        prev = a * float(out[i]) + (1.0 - a) * prev
        out[i] = prev
    return pd.Series(out, index=values.index)


def apply_horizon_strategy(
    timestamp: pd.Series,
    is_forecast: pd.Series,
    p10: pd.Series,
    p50: pd.Series,
    p90: pd.Series,
    h1_months: int,
    h2_months: int,
    mid_blend_max: float,
    long_blend: float,
    long_blend_growth_per_year: float,
    blend_max: float,
    uncertainty_growth_per_year: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    ts = pd.to_datetime(timestamp)
    fc = is_forecast.astype(bool).to_numpy()
    out10 = p10.astype(float).to_numpy().copy()
    out50 = p50.astype(float).to_numpy().copy()
    out90 = p90.astype(float).to_numpy().copy()
    blend_w = np.zeros(len(ts), dtype=float)
    lead_m = np.zeros(len(ts), dtype=int)

    if not fc.any():
        return (
            pd.Series(out10, index=p10.index),
            pd.Series(out50, index=p50.index),
            pd.Series(out90, index=p90.index),
            pd.Series(blend_w, index=p50.index),
            pd.Series(lead_m, index=p50.index),
        )

    h1 = int(max(1, h1_months))
    h2 = int(max(h1 + 1, h2_months))
    w_mid = float(np.clip(mid_blend_max, 0.0, 0.95))
    w_long = float(np.clip(long_blend, 0.0, 0.95))
    w_growth = float(max(0.0, long_blend_growth_per_year))
    w_cap = float(np.clip(blend_max, 0.0, 0.95))
    ug = float(max(0.0, uncertainty_growth_per_year))

    # Horizon literature suggests climatology blending for longer lead times.
    hist_mask = ~fc
    if hist_mask.any():
        df_hist = pd.DataFrame({"ts": ts[hist_mask], "y": out50[hist_mask]})
        month_clim = df_hist.groupby(df_hist["ts"].dt.month)["y"].mean().to_dict()
        global_mean = float(df_hist["y"].mean())
    else:
        month_clim = {}
        global_mean = float(np.nanmean(out50))

    first_fc_idx = int(np.argmax(fc))
    first_fc_ts = ts.iloc[first_fc_idx]

    for i in range(first_fc_idx, len(ts)):
        if not fc[i]:
            continue
        lm = (ts.iloc[i].year - first_fc_ts.year) * 12 + (ts.iloc[i].month - first_fc_ts.month) + 1
        lm = int(max(1, lm))
        lead_m[i] = lm

        if lm <= h1:
            w = 0.0
        elif lm <= h2:
            frac = (lm - h1) / max(1, (h2 - h1))
            w = w_mid * frac
        else:
            extra_years = (lm - h2) / 12.0
            w = min(w_cap, w_long + w_growth * extra_years)
        w = float(np.clip(w, 0.0, w_cap))
        blend_w[i] = w

        m = int(ts.iloc[i].month)
        clim = float(month_clim.get(m, global_mean))
        out50[i] = (1.0 - w) * out50[i] + w * clim

        # Uncertainty typically grows with lead time.
        grow = 1.0 + ug * (lm / 12.0)
        lo_w = max(0.0, out50[i] - out10[i]) * grow
        hi_w = max(0.0, out90[i] - out50[i]) * grow
        out10[i] = max(0.0, out50[i] - lo_w)
        out90[i] = max(out50[i], out50[i] + hi_w)

    return (
        pd.Series(out10, index=p10.index),
        pd.Series(out50, index=p50.index),
        pd.Series(out90, index=p90.index),
        pd.Series(blend_w, index=p50.index),
        pd.Series(lead_m, index=p50.index),
    )


def build_input_diagnostics(merged: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(merged["timestamp"])
    fc = merged["is_forecast"].astype(bool)
    out: dict[str, Any] = {"forecast_rows": int(fc.sum()), "variables": {}, "warnings": []}
    for var in ["temp", "humidity", "precip", "pressure"]:
        v = merged[var].astype(float)
        vf = v[fc]
        rows = int(len(vf))
        if rows < 12:
            out["variables"][var] = {
                "forecast_rows": rows,
                "forecast_std": float(vf.std(ddof=0)) if rows else None,
                "month_to_month_std_mean": None,
                "seasonal_amplitude": None,
                "repeat_ratio": None,
                "low_variability_warning": bool(rows > 0),
            }
            if rows > 0:
                out["warnings"].append(f"{var}:limited_forecast_length_{rows}")
            continue

        mf = pd.DataFrame({"ts": ts[fc], "v": vf})
        month_stats = mf.groupby(mf["ts"].dt.month)["v"].std(ddof=0)
        month_means = mf.groupby(mf["ts"].dt.month)["v"].mean()
        month_std_mean = float(month_stats.mean()) if len(month_stats) else 0.0
        seasonal_amp = float(month_means.max() - month_means.min()) if len(month_means) else 0.0
        forecast_std = float(vf.std(ddof=0))
        ratio = float(month_std_mean / max(seasonal_amp, 1e-6))
        low_var = bool((month_std_mean < 1e-6) or (ratio < 0.01))

        out["variables"][var] = {
            "forecast_rows": rows,
            "forecast_std": forecast_std,
            "month_to_month_std_mean": month_std_mean,
            "seasonal_amplitude": seasonal_amp,
            "repeat_ratio": ratio,
            "low_variability_warning": low_var,
        }
        if low_var:
            out["warnings"].append(f"{var}:near_repeat_forecast_pattern")
    return out


def generate_stochastic_realization(
    p10: pd.Series,
    p50: pd.Series,
    p90: pd.Series,
    is_forecast: pd.Series,
    random_seed: int,
    ar1_rho: float,
    scale: float,
) -> pd.Series:
    y10 = p10.astype(float).to_numpy()
    y50 = p50.astype(float).to_numpy()
    y90 = p90.astype(float).to_numpy()
    fc = is_forecast.astype(bool).to_numpy()
    out = y50.copy()
    if (not fc.any()) or float(scale) <= 0:
        return pd.Series(out, index=p50.index)

    z80 = 1.2815515655446004
    base_sigma = np.maximum((y90 - y10) / max(2.0 * z80, 1e-6), 0.02)
    target_sigma = np.maximum(0.0, float(scale)) * base_sigma
    rho = float(np.clip(ar1_rho, -0.95, 0.95))
    innov_scale = math.sqrt(max(1e-8, 1.0 - rho * rho))

    rng = np.random.default_rng(int(random_seed) + 7919)
    start = int(np.argmax(fc))
    anom_prev = 0.0
    for i in range(start, len(out)):
        if not fc[i]:
            anom_prev = 0.0
            continue
        eps = float(rng.normal(0.0, target_sigma[i] * innov_scale))
        anom = rho * anom_prev + eps
        y = y50[i] + anom
        y = float(np.clip(y, max(0.0, y10[i]), max(y10[i], y90[i])))
        out[i] = y
        anom_prev = y - y50[i]

    return pd.Series(out, index=p50.index)


def compute_driver_sensitivity(
    merged: pd.DataFrame,
    norm_bounds: dict[str, tuple[float, float]],
    temp_opt_c: float,
    temp_penalty_per_c: float,
    elevation_m: float,
    gamma_pdc: float,
    calib: CalibrationInfo,
) -> dict[str, Any]:
    cloud_extra = (
        merged["cloud_cover_internet_pct"].astype(float).to_numpy()
        if "cloud_cover_internet_pct" in merged.columns
        else None
    )
    wind_extra = (
        merged["wind_speed_internet_kmh"].astype(float).to_numpy()
        if "wind_speed_internet_kmh" in merged.columns
        else None
    )
    shortwave_extra = (
        merged["shortwave_internet_kwh_m2_day"].astype(float).to_numpy()
        if "shortwave_internet_kwh_m2_day" in merged.columns
        else None
    )
    base_comp = compute_heuristic_components(
        temp=merged["temp"].astype(float).to_numpy(),
        humidity=merged["humidity"].astype(float).to_numpy(),
        precip=merged["precip"].astype(float).to_numpy(),
        pressure=merged["pressure"].astype(float).to_numpy(),
        extra_radiation_kwh=merged["extra_radiation_kwh_m2_day"].astype(float).to_numpy(),
        temp_opt_c=float(temp_opt_c),
        temp_penalty_per_c=float(temp_penalty_per_c),
        elevation_m=float(elevation_m),
        gamma_pdc=float(gamma_pdc),
        norm_bounds=norm_bounds,
        cloud_cover_pct=cloud_extra,
        wind_speed_kmh=wind_extra,
        shortwave_kwh=shortwave_extra,
    )
    base_pred = np.clip(base_comp["heuristic_kwh"].astype(float), 0.0, None)
    if calib.enabled:
        base_pred = np.clip(base_pred * float(calib.slope) + float(calib.intercept), 0.0, None)
    base_mean = float(np.mean(base_pred)) if len(base_pred) else 0.0

    hist = merged[~merged["is_forecast"]].copy()
    if len(hist) < 3:
        hist = merged.copy()

    var_defs = [
        ("temp", -50.0, 65.0),
        ("humidity", 0.0, 100.0),
        ("precip", 0.0, float("inf")),
        ("pressure", 0.0, float("inf")),
    ]

    rows: dict[str, Any] = {}
    for var, lo, hi in var_defs:
        std = float(hist[var].astype(float).std(ddof=0))
        if (not np.isfinite(std)) or std <= 1e-9:
            std = float(SIGMA_MIN[var])
        delta = max(float(SIGMA_MIN[var]), 0.80 * std)
        base_arr = merged[var].astype(float).to_numpy()
        plus = np.clip(base_arr + delta, lo, hi)
        minus = np.clip(base_arr - delta, lo, hi)

        inp_plus = {
            "temp": merged["temp"].astype(float).to_numpy().copy(),
            "humidity": merged["humidity"].astype(float).to_numpy().copy(),
            "precip": merged["precip"].astype(float).to_numpy().copy(),
            "pressure": merged["pressure"].astype(float).to_numpy().copy(),
        }
        inp_minus = {k: v.copy() for k, v in inp_plus.items()}
        inp_plus[var] = plus
        inp_minus[var] = minus

        c_plus = compute_heuristic_components(
            temp=inp_plus["temp"],
            humidity=inp_plus["humidity"],
            precip=inp_plus["precip"],
            pressure=inp_plus["pressure"],
            extra_radiation_kwh=merged["extra_radiation_kwh_m2_day"].astype(float).to_numpy(),
            temp_opt_c=float(temp_opt_c),
            temp_penalty_per_c=float(temp_penalty_per_c),
            elevation_m=float(elevation_m),
            gamma_pdc=float(gamma_pdc),
            norm_bounds=norm_bounds,
            cloud_cover_pct=cloud_extra,
            wind_speed_kmh=wind_extra,
            shortwave_kwh=shortwave_extra,
        )
        c_minus = compute_heuristic_components(
            temp=inp_minus["temp"],
            humidity=inp_minus["humidity"],
            precip=inp_minus["precip"],
            pressure=inp_minus["pressure"],
            extra_radiation_kwh=merged["extra_radiation_kwh_m2_day"].astype(float).to_numpy(),
            temp_opt_c=float(temp_opt_c),
            temp_penalty_per_c=float(temp_penalty_per_c),
            elevation_m=float(elevation_m),
            gamma_pdc=float(gamma_pdc),
            norm_bounds=norm_bounds,
            cloud_cover_pct=cloud_extra,
            wind_speed_kmh=wind_extra,
            shortwave_kwh=shortwave_extra,
        )

        y_plus = np.clip(c_plus["heuristic_kwh"].astype(float), 0.0, None)
        y_minus = np.clip(c_minus["heuristic_kwh"].astype(float), 0.0, None)
        if calib.enabled:
            y_plus = np.clip(y_plus * float(calib.slope) + float(calib.intercept), 0.0, None)
            y_minus = np.clip(y_minus * float(calib.slope) + float(calib.intercept), 0.0, None)

        signed = 0.5 * (y_plus - y_minus)
        abs_eff = np.abs(signed)
        mean_abs = float(np.mean(abs_eff))
        mean_signed = float(np.mean(signed))
        rel_pct = float(100.0 * mean_abs / max(base_mean, 1e-6))
        direction = "increase" if mean_signed > 0 else "decrease" if mean_signed < 0 else "neutral"

        rows[var] = {
            "delta_used": float(delta),
            "mean_abs_effect_kwh_m2_day": mean_abs,
            "mean_signed_effect_kwh_m2_day": mean_signed,
            "relative_abs_effect_pct_of_mean": rel_pct,
            "direction_with_increase": direction,
        }

    ranking = sorted(rows.keys(), key=lambda k: rows[k]["mean_abs_effect_kwh_m2_day"], reverse=True)
    return {
        "method": "one_sigma_central_difference",
        "base_mean_kwh_m2_day": float(base_mean),
        "variables": rows,
        "ranking_by_abs_effect": ranking,
    }


def plot_solar_chart(
    out: pd.DataFrame,
    solar_obs: pd.DataFrame,
    driver_sensitivity: dict[str, Any],
    output_chart: Path,
) -> str | None:
    if not MPL_OK:
        return "matplotlib_not_available"
    try:
        df = out.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ts = df["timestamp"]
        hist = df[~df["is_forecast"]].copy()
        fc = df[df["is_forecast"]].copy()
        fc_start = pd.to_datetime(fc["timestamp"]).min() if len(fc) else None

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.30, 1.0], hspace=0.16)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.fill_between(
            ts,
            df["solar_potential_p10_kwh_m2_day"].astype(float),
            df["solar_potential_p90_kwh_m2_day"].astype(float),
            color="#f59e0b",
            alpha=0.22,
            label="P10-P90 uncertainty",
            linewidth=0,
        )
        ax1.plot(
            ts,
            df["solar_potential_p50_kwh_m2_day"].astype(float),
            color="#b91c1c",
            linewidth=2.2,
            label="Solar expected (P50)",
        )
        if "solar_potential_scenario_kwh_m2_day" in df.columns:
            ax1.plot(
                ts,
                df["solar_potential_scenario_kwh_m2_day"].astype(float),
                color="#1d4ed8",
                linewidth=1.1,
                linestyle="--",
                alpha=0.70,
                label="Solar realization",
            )

        if len(solar_obs):
            obs = solar_obs.copy()
            obs["timestamp"] = pd.to_datetime(obs["timestamp"])
            obs = obs.merge(df[["timestamp"]], on="timestamp", how="inner")
            if len(obs):
                ax1.scatter(
                    obs["timestamp"],
                    obs["solar_obs"].astype(float),
                    color="#111827",
                    s=24,
                    alpha=0.85,
                    label="Observed solar",
                )

        if fc_start is not None:
            ax1.axvline(fc_start, color="#6b7280", linestyle="--", linewidth=1.0, label="Forecast start")
            ax1.axvspan(fc_start, ts.max(), color="#e5e7eb", alpha=0.24)

        ax1.set_ylabel("kWh/m2/day")
        ax1.set_title("Solar Potential Forecast (Physics + Weather + ML)")
        ax1.grid(alpha=0.25)
        ax1.legend(ncol=3, fontsize=9, frameon=False, loc="upper left")

        # Lower panel: compact regime indicators for readability.
        util = pd.to_numeric(df["clear_sky_utilization_ratio"], errors="coerce") if "clear_sky_utilization_ratio" in df.columns else None
        if util is not None and int(util.notna().sum()) >= 2:
            util_s = util.rolling(3, min_periods=1).mean().clip(lower=0.0, upper=1.2)
            ax2.plot(ts, util_s, color="#0f766e", linewidth=1.7, label="Clear-sky utilization")
        if "cloudiness_percent" in df.columns:
            cloud = (pd.to_numeric(df["cloudiness_percent"], errors="coerce") / 100.0).clip(lower=0.0, upper=1.0)
            cloud_s = cloud.rolling(3, min_periods=1).mean()
            ax2.plot(ts, cloud_s, color="#1d4ed8", linewidth=1.5, linestyle="--", label="Cloudiness (scaled)")
        if "clearness_index" in df.columns:
            kt = pd.to_numeric(df["clearness_index"], errors="coerce").clip(lower=0.0, upper=1.2)
            kt_s = kt.rolling(3, min_periods=1).mean()
            ax2.plot(ts, kt_s, color="#9a3412", linewidth=1.3, alpha=0.85, label="Clearness index")

        ax2.axhline(0.5, color="#9ca3af", linewidth=0.8, alpha=0.6)
        if fc_start is not None:
            ax2.axvspan(fc_start, ts.max(), color="#e5e7eb", alpha=0.24)
        ax2.set_ylim(0.0, 1.15)
        ax2.set_ylabel("ratio")
        ax2.set_xlabel("Date")
        ax2.grid(alpha=0.22)
        ax2.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")

        ranking = driver_sensitivity.get("ranking_by_abs_effect", [])
        if ranking:
            txt = "Driver impact rank: " + " > ".join(ranking)
            ax2.text(0.01, 0.02, txt, transform=ax2.transAxes, fontsize=8, color="#111827")

        output_chart.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_chart, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return None
    except Exception as exc:
        try:
            plt.close("all")
        except Exception:
            pass
        return f"plot_failed:{exc}"


def plot_driver_panel_chart(out: pd.DataFrame, output_chart: Path) -> str | None:
    if not MPL_OK:
        return "matplotlib_not_available"
    try:
        df = out.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ts = df["timestamp"]
        fc = df["is_forecast"].astype(bool) if "is_forecast" in df.columns else pd.Series(False, index=df.index)
        fc_start = pd.to_datetime(df.loc[fc, "timestamp"]).min() if bool(fc.any()) else None
        hist = df[~fc].copy()
        base = hist if len(hist) >= 24 else df

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        cfg = [
            ("temp", "Temperature anomaly", "#b91c1c"),
            ("humidity", "Humidity anomaly", "#2563eb"),
            ("precip", "Precipitation anomaly", "#0f766e"),
            ("pressure", "Pressure anomaly", "#7c3aed"),
        ]

        for ax, (col, title, color) in zip(axes.ravel(), cfg):
            if col not in df.columns:
                ax.set_title(f"{title} (missing)")
                ax.axis("off")
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            mu = float(pd.to_numeric(base[col], errors="coerce").mean())
            sd = float(pd.to_numeric(base[col], errors="coerce").std(ddof=0))
            if (not np.isfinite(sd)) or sd <= 1e-9:
                sd = 1.0
            z = (s - mu) / sd
            z = z.clip(-3.0, 3.0)
            z_s = z.rolling(3, min_periods=1).mean()

            ax.plot(ts, z, color=color, linewidth=0.9, alpha=0.35, label="Monthly z")
            ax.plot(ts, z_s, color=color, linewidth=1.8, label="3-mo smooth z")
            ax.axhline(0.0, color="#9ca3af", linewidth=0.8, alpha=0.7)
            if fc_start is not None:
                ax.axvline(fc_start, color="#6b7280", linestyle="--", linewidth=0.9)
                ax.axvspan(fc_start, ts.max(), color="#e5e7eb", alpha=0.22)
            ax.set_title(title)
            ax.set_ylabel("z-score")
            ax.grid(alpha=0.22)
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        axes[1, 0].set_xlabel("Date")
        axes[1, 1].set_xlabel("Date")
        fig.suptitle("Climate Driver Panels (all core variables)", fontsize=13)
        output_chart.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_chart, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return None
    except Exception as exc:
        try:
            plt.close("all")
        except Exception:
            pass
        return f"plot_failed:{exc}"


def export_separated_outputs(out: pd.DataFrame, csv_dir: Path, chart_dir: Path) -> dict[str, Any]:
    def dedupe_keep_order(cols: list[str]) -> list[str]:
        seen: set[str] = set()
        out_cols: list[str] = []
        for c in cols:
            if (c in seen) or (c not in df.columns):
                continue
            seen.add(c)
            out_cols.append(c)
        return out_cols

    def plot_block(tbl: pd.DataFrame, y_cols: list[str], title: str, ylabel: str, chart_path: Path) -> str:
        if not MPL_OK:
            return "matplotlib_not_available"
        try:
            ts = pd.to_datetime(tbl["timestamp"], errors="coerce")
            fig, ax = plt.subplots(figsize=(13, 4.2))
            palette = [
                "#b91c1c",
                "#1d4ed8",
                "#0f766e",
                "#7c3aed",
                "#9a3412",
                "#0f172a",
                "#475569",
            ]
            plotted = 0
            for i, col in enumerate(y_cols):
                if col not in tbl.columns:
                    continue
                y = pd.to_numeric(tbl[col], errors="coerce")
                if int(y.notna().sum()) < 2:
                    continue
                ax.plot(ts, y, linewidth=1.5, color=palette[i % len(palette)], label=col)
                plotted += 1

            if plotted == 0:
                plt.close(fig)
                return "no_plottable_series"

            if "is_forecast" in tbl.columns:
                fc = tbl["is_forecast"].astype(bool)
                if bool(fc.any()):
                    fc_start = pd.to_datetime(tbl.loc[fc, "timestamp"]).min()
                    ax.axvline(fc_start, color="#6b7280", linestyle="--", linewidth=1.0)
                    ax.axvspan(fc_start, ts.max(), color="#e5e7eb", alpha=0.24)

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Date")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, ncol=3, fontsize=8, loc="best")
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(chart_path, dpi=165, bbox_inches="tight")
            plt.close(fig)
            return "ok"
        except Exception as exc:
            try:
                plt.close("all")
            except Exception:
                pass
            return f"plot_failed:{exc}"

    df = out.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    csv_dir = Path(csv_dir)
    chart_dir = Path(chart_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    common = ["timestamp", "is_forecast", "set", "forecast_lead_month", "model_version"]
    blocks = [
        (
            "solar",
            [
                "solar_potential_kwh_m2_day",
                "solar_potential_expected_kwh_m2_day",
                "solar_potential_scenario_kwh_m2_day",
                "solar_potential_p10_kwh_m2_day",
                "solar_potential_p50_kwh_m2_day",
                "solar_potential_p90_kwh_m2_day",
                "solar_potential_index",
                "solar_potential_class",
            ],
            "kWh/m2/day",
            "Solar Potential (Separated)",
            [
                "solar_potential_p10_kwh_m2_day",
                "solar_potential_p50_kwh_m2_day",
                "solar_potential_p90_kwh_m2_day",
                "solar_potential_scenario_kwh_m2_day",
                "solar_potential_kwh_m2_day",
            ],
        ),
        (
            "temperature",
            [
                "temperature_c",
                "temperature_local_c",
                "temperature_model_c",
                "temperature_internet_c",
                "temperature_internet_aligned_c",
                "temperature_delta_c",
                "temperature_delta_raw_c",
            ],
            "degC",
            "Temperature Channels (Separated)",
            [
                "temperature_local_c",
                "temperature_model_c",
                "temperature_internet_c",
                "temperature_internet_aligned_c",
                "temperature_delta_c",
            ],
        ),
        (
            "humidity",
            [
                "humidity",
                "humidity_local_pct",
                "humidity_model_pct",
                "humidity_internet_pct",
                "humidity_internet_pct_aligned",
                "humidity_delta_pct",
                "humidity_delta_raw_pct",
            ],
            "percent",
            "Humidity Channels (Separated)",
            [
                "humidity_local_pct",
                "humidity_model_pct",
                "humidity_internet_pct",
                "humidity_internet_pct_aligned",
                "humidity_delta_pct",
            ],
        ),
        (
            "precip",
            [
                "precip",
                "precip_local_mm",
                "precip_model_mm",
                "precip_internet_mm_day",
                "precip_internet_mm_month",
                "precip_internet_mm_month_aligned",
                "precip_delta_mm",
                "precip_delta_raw_mm",
            ],
            "mm",
            "Precipitation Channels (Separated)",
            [
                "precip_local_mm",
                "precip_model_mm",
                "precip_internet_mm_month",
                "precip_internet_mm_month_aligned",
                "precip_delta_mm",
            ],
        ),
        (
            "pressure",
            [
                "pressure",
                "pressure_local",
                "pressure_model",
                "pressure_internet",
                "pressure_internet_aligned",
                "pressure_delta",
                "pressure_delta_raw",
            ],
            "pressure units",
            "Pressure Channels (Separated)",
            [
                "pressure_local",
                "pressure_model",
                "pressure_internet",
                "pressure_internet_aligned",
                "pressure_delta_raw",
            ],
        ),
    ]

    meta: dict[str, Any] = {
        "enabled": True,
        "csv_dir": str(csv_dir),
        "chart_dir": str(chart_dir),
        "csv_files": {},
        "chart_files": {},
        "chart_status": {},
        "rows": {},
    }

    for name, cols, ylabel, title, plot_cols in blocks:
        use_cols = dedupe_keep_order(common + cols)
        if len(use_cols) < 2:
            continue
        tbl = df[use_cols].copy()
        csv_path = csv_dir / f"{name}_separated.csv"
        tbl.to_csv(csv_path, index=False)
        meta["csv_files"][name] = str(csv_path)
        meta["rows"][name] = int(len(tbl))

        chart_path = chart_dir / f"{name}_separated.png"
        status = plot_block(tbl, plot_cols, title=title, ylabel=ylabel, chart_path=chart_path)
        meta["chart_files"][name] = str(chart_path)
        meta["chart_status"][name] = status

    return meta


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    return value


def build_quality_checks(out: pd.DataFrame) -> dict[str, Any]:
    y = out["solar_potential_kwh_m2_day"].astype(float)
    p10 = out["solar_potential_p10_kwh_m2_day"].astype(float)
    p50 = out["solar_potential_p50_kwh_m2_day"].astype(float)
    p90 = out["solar_potential_p90_kwh_m2_day"].astype(float)
    extra = out["extra_radiation_kwh_m2_day"].astype(float)
    util = out["clear_sky_utilization_ratio"].astype(float)

    q_order_bad = int(((p10 > p50) | (p50 > p90)).sum())
    neg_count = int((y < 0).sum())
    util_over_1 = int((util > 1.0).sum())
    util_over_0_9 = int((util > 0.90).sum())
    frac_zero = float((y <= 1e-9).mean())
    delta = y.diff().abs().dropna()
    diff_frac_bad = 0
    beam_diff_balance_mae = None
    if "diffuse_fraction" in out.columns:
        dfc = out["diffuse_fraction"].astype(float)
        diff_frac_bad = int(((dfc < 0) | (dfc > 1)).sum())
    if all(c in out.columns for c in ["beam_kwh_m2_day", "diffuse_kwh_m2_day", "global_horizontal_kwh_m2_day"]):
        rec = out["beam_kwh_m2_day"].astype(float) + out["diffuse_kwh_m2_day"].astype(float)
        ghi = out["global_horizontal_kwh_m2_day"].astype(float)
        beam_diff_balance_mae = float(np.mean(np.abs(rec - ghi)))

    return {
        "quantile_order_violations": q_order_bad,
        "negative_prediction_count": neg_count,
        "utilization_over_1_count": util_over_1,
        "utilization_over_0_9_count": util_over_0_9,
        "diffuse_fraction_out_of_bounds_count": diff_frac_bad,
        "beam_diffuse_balance_mae": beam_diff_balance_mae,
        "zero_value_fraction": frac_zero,
        "monthly_abs_change_mean": float(delta.mean()) if not delta.empty else 0.0,
        "monthly_abs_change_p95": float(delta.quantile(0.95)) if not delta.empty else 0.0,
        "solar_over_extraterrestrial_max_ratio": float(np.nanmax(np.divide(y, np.maximum(extra, 1e-9)))),
    }


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return None
    c = float(x[m].corr(y[m]))
    return c if np.isfinite(c) else None


def build_internet_consistency(out: pd.DataFrame) -> dict[str, Any]:
    required = {
        "temperature_c",
        "temperature_internet_c",
        "cloudiness_percent",
        "cloud_cover_internet_pct",
        "global_horizontal_kwh_m2_day",
        "shortwave_internet_kwh_m2_day",
    }
    if not required.issubset(set(out.columns)):
        return {"available": False, "reason": "missing_columns"}

    def choose_reference(df: pd.DataFrame, raw_col: str, aligned_col: str, min_points: int = 3) -> str:
        if aligned_col in df.columns:
            n_valid = int(pd.to_numeric(df[aligned_col], errors="coerce").notna().sum())
            if n_valid >= int(min_points):
                return aligned_col
        return raw_col

    def block(df: pd.DataFrame) -> dict[str, Any]:
        if len(df) < 3:
            return {"rows": int(len(df)), "available": False}
        t_local = df["temperature_c"].astype(float)
        t_net_col = choose_reference(df, "temperature_internet_c", "temperature_internet_aligned_c")
        t_net = df[t_net_col].astype(float)
        c_proxy = df["cloudiness_percent"].astype(float)
        c_net = df["cloud_cover_internet_pct"].astype(float)
        ghi = df["global_horizontal_kwh_m2_day"].astype(float)
        sw = df["shortwave_internet_kwh_m2_day"].astype(float)
        t_delta = t_local - t_net
        c_delta = c_proxy - c_net
        return {
            "rows": int(len(df)),
            "available": True,
            "temp_bias_c_mean": float(t_delta.mean()),
            "temp_mae_c": float(np.mean(np.abs(t_delta))),
            "temp_corr": _safe_corr(t_local, t_net),
            "cloud_bias_pct_mean": float(c_delta.mean()),
            "cloud_mae_pct": float(np.mean(np.abs(c_delta))),
            "cloud_corr": _safe_corr(c_proxy, c_net),
            "radiation_corr": _safe_corr(ghi, sw),
            "temperature_reference": t_net_col,
        }

    def add_optional(block_out: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        if {"humidity_model_pct", "humidity_internet_pct"}.issubset(set(df.columns)):
            h_loc = df["humidity_model_pct"].astype(float)
            h_net_col = choose_reference(df, "humidity_internet_pct", "humidity_internet_pct_aligned")
            h_net = df[h_net_col].astype(float)
            h_delta = h_loc - h_net
            block_out["humidity_bias_pct_mean"] = float(h_delta.mean())
            block_out["humidity_mae_pct"] = float(np.mean(np.abs(h_delta)))
            block_out["humidity_corr"] = _safe_corr(h_loc, h_net)
            block_out["humidity_reference"] = h_net_col
        if {"precip_model_mm", "precip_internet_mm_month"}.issubset(set(df.columns)):
            p_loc = df["precip_model_mm"].astype(float)
            p_net_col = choose_reference(df, "precip_internet_mm_month", "precip_internet_mm_month_aligned")
            p_net = df[p_net_col].astype(float)
            p_delta = p_loc - p_net
            block_out["precip_bias_mm_mean"] = float(p_delta.mean())
            block_out["precip_mae_mm"] = float(np.mean(np.abs(p_delta)))
            block_out["precip_corr"] = _safe_corr(p_loc, p_net)
            block_out["precip_reference"] = p_net_col
        if {"pressure_model", "pressure_internet"}.issubset(set(df.columns)):
            ps_loc = df["pressure_model"].astype(float)
            ps_net_col = choose_reference(df, "pressure_internet", "pressure_internet_aligned")
            ps_net = df[ps_net_col].astype(float)
            m = ps_loc.notna() & ps_net.notna()
            if int(m.sum()) >= 3:
                loc_m = ps_loc[m]
                net_m = ps_net[m]
                loc_med = float(np.nanmedian(np.abs(loc_m)))
                net_med = float(np.nanmedian(np.abs(net_m)))
                scale_ratio = float(net_med / max(1e-9, loc_med))
                if (scale_ratio < 0.50) or (scale_ratio > 2.00):
                    block_out["pressure_bias_mean"] = None
                    block_out["pressure_mae"] = None
                    block_out["pressure_corr"] = None
                    block_out["pressure_comparable_scale"] = False
                    block_out["pressure_scale_ratio"] = scale_ratio
                else:
                    ps_delta = loc_m - net_m
                    block_out["pressure_bias_mean"] = float(ps_delta.mean())
                    block_out["pressure_mae"] = float(np.mean(np.abs(ps_delta)))
                    block_out["pressure_corr"] = _safe_corr(loc_m, net_m)
                    block_out["pressure_comparable_scale"] = True
                    block_out["pressure_scale_ratio"] = scale_ratio
            else:
                block_out["pressure_bias_mean"] = None
                block_out["pressure_mae"] = None
                block_out["pressure_corr"] = None
                block_out["pressure_comparable_scale"] = False
                block_out["pressure_scale_ratio"] = None
            block_out["pressure_reference"] = ps_net_col
        return block_out

    fc = out[out["is_forecast"].astype(bool)].copy()
    hist = out[~out["is_forecast"].astype(bool)].copy()
    all_block = add_optional(block(out), out)
    hist_block = add_optional(block(hist), hist)
    fc_block = add_optional(block(fc), fc)
    return {
        "available": True,
        "all": all_block,
        "history": hist_block,
        "forecast": fc_block,
    }


def main() -> None:
    args = parse_args()
    if not (0 <= args.clip_low_q < args.clip_high_q <= 1):
        raise SystemExit("clip quantiles must satisfy 0 <= low < high <= 1")
    if not (0 <= args.low_point_calibration_shrink <= 1):
        raise SystemExit("--low-point-calibration-shrink must be between 0 and 1")
    if int(args.min_calibration_points) < 1:
        raise SystemExit("--min-calibration-points must be >= 1")
    if args.mc_samples < 50:
        raise SystemExit("--mc-samples must be >= 50")
    if args.uncertainty_z <= 0:
        raise SystemExit("--uncertainty-z must be > 0")
    if not (0 <= args.forecast_smoothing_alpha <= 1):
        raise SystemExit("--forecast-smoothing-alpha must be between 0 and 1")
    if int(args.horizon_h1_months) < 1:
        raise SystemExit("--horizon-h1-months must be >= 1")
    if int(args.horizon_h2_months) <= int(args.horizon_h1_months):
        raise SystemExit("--horizon-h2-months must be > horizon-h1-months")
    if not (0 <= args.horizon_mid_blend_max <= 1):
        raise SystemExit("--horizon-mid-blend-max must be between 0 and 1")
    if not (0 <= args.horizon_long_blend <= 1):
        raise SystemExit("--horizon-long-blend must be between 0 and 1")
    if float(args.horizon_long_blend_growth_per_year) < 0:
        raise SystemExit("--horizon-long-blend-growth-per-year must be >= 0")
    if not (0 <= args.horizon_blend_max <= 1):
        raise SystemExit("--horizon-blend-max must be between 0 and 1")
    if float(args.horizon_blend_max) < float(args.horizon_long_blend):
        raise SystemExit("--horizon-blend-max must be >= --horizon-long-blend")
    if float(args.uncertainty_growth_per_year) < 0:
        raise SystemExit("--uncertainty-growth-per-year must be >= 0")
    if not (-0.95 <= float(args.scenario_ar1_rho) <= 0.95):
        raise SystemExit("--scenario-ar1-rho must be in [-0.95, 0.95]")
    if float(args.scenario_scale) < 0:
        raise SystemExit("--scenario-scale must be >= 0")
    if int(args.min_history_for_horizon_blend) < 1:
        raise SystemExit("--min-history-for-horizon-blend must be >= 1")
    if not np.isfinite(args.elevation_m):
        raise SystemExit("--elevation-m must be finite")
    if not np.isfinite(args.latitude) or not np.isfinite(args.longitude):
        raise SystemExit("--latitude and --longitude must be finite")
    if int(args.internet_timeout_seconds) < 5:
        raise SystemExit("--internet-timeout-seconds must be >= 5")
    if not (0 <= float(args.forecast_temp_internet_blend) <= 1):
        raise SystemExit("--forecast-temp-internet-blend must be between 0 and 1")
    if not (0 <= float(args.forecast_humidity_internet_blend) <= 1):
        raise SystemExit("--forecast-humidity-internet-blend must be between 0 and 1")
    if not (0 <= float(args.forecast_precip_internet_blend) <= 1):
        raise SystemExit("--forecast-precip-internet-blend must be between 0 and 1")
    if not (0 <= float(args.forecast_pressure_internet_blend) <= 1):
        raise SystemExit("--forecast-pressure-internet-blend must be between 0 and 1")
    if not (0 <= float(args.forecast_shortwave_internet_blend) <= 1):
        raise SystemExit("--forecast-shortwave-internet-blend must be between 0 and 1")
    if not (0 <= float(args.forecast_cloudiness_internet_blend) <= 1):
        raise SystemExit("--forecast-cloudiness-internet-blend must be between 0 and 1")
    if float(args.assimilation_lead_decay_half_life_months) < 0:
        raise SystemExit("--assimilation-lead-decay-half-life-months must be >= 0")
    if not (0 <= float(args.assimilation_lead_decay_min_factor) <= 1):
        raise SystemExit("--assimilation-lead-decay-min-factor must be between 0 and 1")

    gamma_pdc = resolve_gamma_pdc(args.pv_module_type, float(args.gamma_pdc))

    merged, source_meta = select_source_and_load(args)

    # Fill small temporal holes.
    idx = merged.set_index("timestamp")
    for col in ["temp", "humidity", "precip", "pressure"]:
        idx[col] = idx[col].interpolate(method="time").ffill().bfill()
    merged = idx.reset_index()
    merged, internet_extra_meta = attach_internet_extras(merged, args)
    merged, internet_assim_meta = apply_internet_assimilation(merged, args)
    input_diagnostics = build_input_diagnostics(merged)

    geom = calc_solar_geometry(merged["timestamp"], latitude_deg=float(args.latitude))
    merged = pd.concat([merged.reset_index(drop=True), geom.reset_index(drop=True)], axis=1)

    hist = merged[~merged["is_forecast"]].copy()
    base = hist if len(hist) >= 12 else merged.copy()

    temp_lo = float(base["temp"].quantile(args.clip_low_q))
    temp_hi = float(base["temp"].quantile(args.clip_high_q))
    hum_lo = float(base["humidity"].quantile(args.clip_low_q))
    hum_hi = float(base["humidity"].quantile(args.clip_high_q))
    prc_lo = float(base["precip"].quantile(args.clip_low_q))
    prc_hi = float(base["precip"].quantile(args.clip_high_q))
    prs_lo = float(base["pressure"].quantile(args.clip_low_q))
    prs_hi = float(base["pressure"].quantile(args.clip_high_q))

    norm_bounds = {
        "temp": (temp_lo, temp_hi),
        "humidity": (hum_lo, hum_hi),
        "precip": (prc_lo, prc_hi),
        "pressure": (prs_lo, prs_hi),
    }

    comp_det = compute_heuristic_components(
        temp=merged["temp"].astype(float).to_numpy(),
        humidity=merged["humidity"].astype(float).to_numpy(),
        precip=merged["precip"].astype(float).to_numpy(),
        pressure=merged["pressure"].astype(float).to_numpy(),
        extra_radiation_kwh=merged["extra_radiation_kwh_m2_day"].astype(float).to_numpy(),
        temp_opt_c=float(args.temp_opt_c),
        temp_penalty_per_c=float(args.temp_penalty_per_c),
        elevation_m=float(args.elevation_m),
        gamma_pdc=float(gamma_pdc),
        norm_bounds=norm_bounds,
        cloud_cover_pct=merged["cloud_cover_internet_pct"].astype(float).to_numpy()
        if "cloud_cover_internet_pct" in merged.columns
        else None,
        wind_speed_kmh=merged["wind_speed_internet_kmh"].astype(float).to_numpy()
        if "wind_speed_internet_kmh" in merged.columns
        else None,
        shortwave_kwh=merged["shortwave_internet_kwh_m2_day"].astype(float).to_numpy()
        if "shortwave_internet_kwh_m2_day" in merged.columns
        else None,
    )
    merged["heuristic_kwh_m2_day"] = comp_det["heuristic_kwh"].astype(float)

    solar_obs, solar_obs_meta = build_solar_observation_set(merged=merged, args=args)
    overlap = merged[["timestamp", "heuristic_kwh_m2_day"]].merge(solar_obs, on="timestamp", how="inner")
    calib = (
        calibrate_series(
            overlap["heuristic_kwh_m2_day"],
            overlap["solar_obs"],
            low_point_shrink=float(args.low_point_calibration_shrink),
            min_points=int(args.min_calibration_points),
        )
        if len(overlap)
        else CalibrationInfo(False, 0, 1.0, 0.0, float("nan"), float("nan"), "none")
    )
    driver_sensitivity = compute_driver_sensitivity(
        merged=merged,
        norm_bounds=norm_bounds,
        temp_opt_c=float(args.temp_opt_c),
        temp_penalty_per_c=float(args.temp_penalty_per_c),
        elevation_m=float(args.elevation_m),
        gamma_pdc=float(gamma_pdc),
        calib=calib,
    )

    calibrated_kwh = merged["heuristic_kwh_m2_day"].astype(float)
    if calib.enabled:
        calibrated_kwh = np.clip(calibrated_kwh * float(calib.slope) + float(calib.intercept), 0.0, None)

    ml_pred, ml_unc, ml_meta = run_ml_correction(
        full_df=merged.assign(heuristic_kwh_m2_day=calibrated_kwh),
        solar_obs=solar_obs,
        min_months=int(args.ml_min_months),
        holdout=int(args.ml_holdout),
    )

    mode_req = str(args.mode).strip().lower()
    if mode_req == "ml" and ml_pred is None:
        reason = ml_meta.get("reason", "unknown")
        raise SystemExit(f"ML mode requested but unavailable: {reason}")

    use_ml = ml_pred is not None and (mode_req in {"auto", "ml"})
    uncertainty_method = ""
    p10 = pd.Series(index=merged.index, dtype=float)
    p50 = pd.Series(index=merged.index, dtype=float)
    p90 = pd.Series(index=merged.index, dtype=float)

    if use_ml:
        final_kwh = np.clip(ml_pred.astype(float), 0.0, None)
        if ml_unc is not None:
            u = np.clip(ml_unc.astype(float), 0.05, None)
        else:
            u = np.clip(0.12 * np.maximum(final_kwh, 1.0), 0.05, None)
        zq = 1.2815515655446004
        p10_n = np.clip(final_kwh - zq * u, 0.0, None)
        p50 = final_kwh
        p90_n = np.clip(final_kwh + zq * u, 0.0, None)
        conf = ml_meta.get("conformal", {}) if isinstance(ml_meta, dict) else {}
        if bool(conf.get("enabled", False)):
            q10 = float(conf.get("q10_residual"))
            q90 = float(conf.get("q90_residual"))
            p10_c = np.clip(final_kwh + q10, 0.0, None)
            p90_c = np.clip(final_kwh + q90, 0.0, None)
            h_rows = int(ml_meta.get("holdout_rows", 0))
            # Use more conformal weight when holdout sample is large enough.
            w_conf = float(np.clip((h_rows - 8) / 28.0, 0.0, 0.75))
            p10 = np.clip((1.0 - w_conf) * p10_n + w_conf * p10_c, 0.0, None)
            p90 = np.clip((1.0 - w_conf) * p90_n + w_conf * p90_c, 0.0, None)
            ml_meta["interval_blend"] = {
                "method": "normal_plus_conformal",
                "w_conformal": w_conf,
                "w_normal": float(1.0 - w_conf),
            }
        else:
            p10 = p10_n
            p90 = p90_n
            ml_meta["interval_blend"] = {
                "method": "normal_only",
                "w_conformal": 0.0,
                "w_normal": 1.0,
            }
        method = "hybrid_physics_ml_v3"
        uncertainty_method = "ml_normal_conformal_blend"
    else:
        hist_std = {
            "temp": float(base["temp"].astype(float).std(ddof=0)),
            "humidity": float(base["humidity"].astype(float).std(ddof=0)),
            "precip": float(base["precip"].astype(float).std(ddof=0)),
            "pressure": float(base["pressure"].astype(float).std(ddof=0)),
        }

        sigma_map = {
            "temp": derive_sigma_series(
                merged["temp"],
                merged["temp_low"],
                merged["temp_high"],
                variable="temp",
                history_std=hist_std["temp"] if np.isfinite(hist_std["temp"]) else 0.0,
                z=float(args.uncertainty_z),
            ),
            "humidity": derive_sigma_series(
                merged["humidity"],
                merged["humidity_low"],
                merged["humidity_high"],
                variable="humidity",
                history_std=hist_std["humidity"] if np.isfinite(hist_std["humidity"]) else 0.0,
                z=float(args.uncertainty_z),
            ),
            "precip": derive_sigma_series(
                merged["precip"],
                merged["precip_low"],
                merged["precip_high"],
                variable="precip",
                history_std=hist_std["precip"] if np.isfinite(hist_std["precip"]) else 0.0,
                z=float(args.uncertainty_z),
            ),
            "pressure": derive_sigma_series(
                merged["pressure"],
                merged["pressure_low"],
                merged["pressure_high"],
                variable="pressure",
                history_std=hist_std["pressure"] if np.isfinite(hist_std["pressure"]) else 0.0,
                z=float(args.uncertainty_z),
            ),
        }

        p10, p50, p90 = run_monte_carlo(
            merged=merged,
            norm_bounds=norm_bounds,
            calib=calib,
            sigma_map=sigma_map,
            mc_samples=int(args.mc_samples),
            random_seed=int(args.random_seed),
            temp_opt_c=float(args.temp_opt_c),
            temp_penalty_per_c=float(args.temp_penalty_per_c),
            elevation_m=float(args.elevation_m),
            gamma_pdc=float(gamma_pdc),
        )
        final_kwh = p50.astype(float)
        method = "physics_attenuation_mc_v3"
        uncertainty_method = "monte_carlo_weather"
        merged["temp_sigma"] = sigma_map["temp"].astype(float)
        merged["humidity_sigma"] = sigma_map["humidity"].astype(float)
        merged["precip_sigma"] = sigma_map["precip"].astype(float)
        merged["pressure_sigma"] = sigma_map["pressure"].astype(float)

    low = np.clip(p10.astype(float), 0.0, None)
    high = np.clip(p90.astype(float), 0.0, None)
    p50 = p50.astype(float)

    # Stabilize forecast segment only: smooth median and interval widths.
    alpha = float(args.forecast_smoothing_alpha)
    if alpha > 0:
        width_lo = np.maximum(0.0, p50 - low)
        width_hi = np.maximum(0.0, high - p50)
        p50_s = smooth_forecast_only(p50, merged["is_forecast"], alpha=alpha)
        wl_s = smooth_forecast_only(width_lo, merged["is_forecast"], alpha=alpha)
        wh_s = smooth_forecast_only(width_hi, merged["is_forecast"], alpha=alpha)
        p50 = np.clip(p50_s, 0.0, None)
        low = np.clip(p50 - wl_s, 0.0, None)
        high = np.clip(p50 + wh_s, 0.0, None)

    horizon_blend_weight = pd.Series(np.zeros(len(merged)), index=merged.index, dtype=float)
    forecast_lead_month = pd.Series(np.zeros(len(merged), dtype=int), index=merged.index)
    horizon_requested = bool(to_bool_text(args.horizon_aware))
    horizon_applied = False
    horizon_reason = "disabled_by_flag"
    history_months = int((~merged["is_forecast"]).sum())
    if horizon_requested and history_months >= int(args.min_history_for_horizon_blend):
        low, p50, high, horizon_blend_weight, forecast_lead_month = apply_horizon_strategy(
            timestamp=merged["timestamp"],
            is_forecast=merged["is_forecast"],
            p10=low,
            p50=p50,
            p90=high,
            h1_months=int(args.horizon_h1_months),
            h2_months=int(args.horizon_h2_months),
            mid_blend_max=float(args.horizon_mid_blend_max),
            long_blend=float(args.horizon_long_blend),
            long_blend_growth_per_year=float(args.horizon_long_blend_growth_per_year),
            blend_max=float(args.horizon_blend_max),
            uncertainty_growth_per_year=float(args.uncertainty_growth_per_year),
        )
        horizon_applied = True
        horizon_reason = "applied"
    elif horizon_requested:
        horizon_reason = f"insufficient_history_{history_months}"

    # Hard physical cap: do not exceed clear-sky envelope with module gain.
    physical_cap = np.clip(comp_det["clear_sky_kwh"].astype(float) * 1.08, 0.0, None)
    p50 = np.minimum(p50.astype(float), physical_cap)
    low = np.minimum(low.astype(float), p50)
    high = np.maximum(high.astype(float), p50)
    high = np.minimum(high, physical_cap)

    low, p50, high, shortwave_assim_meta = apply_forecast_shortwave_assimilation(
        p10=low,
        p50=p50,
        p90=high,
        physical_cap=physical_cap,
        pv_temp_eff=comp_det["pv_temp_eff"].astype(float),
        shortwave_internet_kwh=merged["shortwave_internet_kwh_m2_day"].astype(float).to_numpy()
        if "shortwave_internet_kwh_m2_day" in merged.columns
        else None,
        is_forecast=merged["is_forecast"],
        blend_weight=float(args.forecast_shortwave_internet_blend),
    )

    expected_kwh = p50.astype(float)
    scenario_requested = bool(to_bool_text(args.scenario_enable))
    scenario_kwh = (
        generate_stochastic_realization(
            p10=low,
            p50=expected_kwh,
            p90=high,
            is_forecast=merged["is_forecast"],
            random_seed=int(args.random_seed),
            ar1_rho=float(args.scenario_ar1_rho),
            scale=float(args.scenario_scale),
        )
        if scenario_requested
        else expected_kwh.copy()
    )
    final_kwh = scenario_kwh.astype(float)
    ref = final_kwh[~merged["is_forecast"]] if (~merged["is_forecast"]).any() else final_kwh
    q05 = float(np.quantile(ref, 0.05))
    q95 = float(np.quantile(ref, 0.95))
    if (not np.isfinite(q95)) or q95 <= q05:
        q05, q95 = float(np.min(final_kwh)), float(np.max(final_kwh) + 1e-6)
    idx = np.clip(100.0 * (final_kwh - q05) / (q95 - q05), 0.0, 100.0)

    clear_sky_kwh = comp_det["clear_sky_kwh"].astype(float)
    global_horizontal_kwh = comp_det["global_horizontal_kwh"].astype(float)
    diffuse_kwh = comp_det["diffuse_kwh"].astype(float)
    beam_kwh = comp_det["beam_kwh"].astype(float)
    diffuse_fraction = comp_det["diffuse_fraction"].astype(float)
    after_cloud_kwh = comp_det["after_cloud_kwh"].astype(float)
    pv_temp_eff = comp_det["pv_temp_eff"].astype(float)
    cloud_loss_score = comp_det["cloud_loss"].astype(float)
    pressure_bonus = comp_det["pressure_bonus"].astype(float)
    clearness_index = comp_det["clearness_index"].astype(float)
    temp_score = comp_det["temp_score"].astype(float)
    cloud_loss_kwh = np.clip(clear_sky_kwh - after_cloud_kwh, 0.0, None)
    temp_effect_kwh = (comp_det["heuristic_kwh"] - after_cloud_kwh).astype(float)
    clear_sky_util = np.divide(
        final_kwh.astype(float),
        np.maximum(merged["extra_radiation_kwh_m2_day"].astype(float).to_numpy(), 1e-6),
    )
    cloudiness_percent = (100.0 * cloud_loss_score).astype(float)
    cloudiness_blend_weight = 0.0
    cloudiness_blend_reason = "internet_cloud_missing_or_weight_zero"
    if ("cloud_cover_internet_pct" in merged.columns) and float(args.forecast_cloudiness_internet_blend) > 0:
        w_cloud = float(np.clip(float(args.forecast_cloudiness_internet_blend), 0.0, 1.0))
        fc_mask = merged["is_forecast"].astype(bool).to_numpy()
        inet_cloud = merged["cloud_cover_internet_pct"].astype(float).to_numpy()
        m = fc_mask & np.isfinite(inet_cloud)
        if np.any(m):
            cloudiness_percent[m] = (1.0 - w_cloud) * cloudiness_percent[m] + w_cloud * inet_cloud[m]
            cloudiness_percent = np.clip(cloudiness_percent, 0.0, 100.0)
            cloudiness_blend_weight = w_cloud
            cloudiness_blend_reason = "applied"

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(merged["timestamp"]),
            "temp": merged["temp"].astype(float),
            "temperature_c": merged["temp"].astype(float),
            "temperature_model_c": merged["temp"].astype(float),
            "temperature_local_c": merged["temp_local"].astype(float) if "temp_local" in merged.columns else merged["temp"].astype(float),
            "humidity": merged["humidity"].astype(float),
            "humidity_model_pct": merged["humidity"].astype(float),
            "humidity_local_pct": merged["humidity_local"].astype(float) if "humidity_local" in merged.columns else merged["humidity"].astype(float),
            "precip": merged["precip"].astype(float),
            "precip_model_mm": merged["precip"].astype(float),
            "precip_local_mm": merged["precip_local"].astype(float) if "precip_local" in merged.columns else merged["precip"].astype(float),
            "pressure": merged["pressure"].astype(float),
            "pressure_model": merged["pressure"].astype(float),
            "pressure_local": merged["pressure_local"].astype(float) if "pressure_local" in merged.columns else merged["pressure"].astype(float),
            "daylight_hours": merged["daylight_hours"].astype(float),
            "extra_radiation_kwh_m2_day": merged["extra_radiation_kwh_m2_day"].astype(float),
            "clear_sky_kwh_m2_day": clear_sky_kwh,
            "global_horizontal_kwh_m2_day": global_horizontal_kwh,
            "beam_kwh_m2_day": beam_kwh,
            "diffuse_kwh_m2_day": diffuse_kwh,
            "diffuse_fraction": diffuse_fraction,
            "pressure_bonus_factor": pressure_bonus,
            "clearness_index": clearness_index,
            "temp_score": temp_score,
            "cloud_loss_score": cloud_loss_score,
            "cloudiness_percent": cloudiness_percent.astype(float),
            "cloud_loss_kwh_m2_day": cloud_loss_kwh,
            "pv_temp_efficiency_factor": pv_temp_eff,
            "temp_effect_kwh_m2_day": temp_effect_kwh,
            "heuristic_kwh_m2_day": merged["heuristic_kwh_m2_day"].astype(float),
            "solar_potential_p10_kwh_m2_day": low.astype(float),
            "solar_potential_p50_kwh_m2_day": p50.astype(float),
            "solar_potential_p90_kwh_m2_day": high.astype(float),
            "solar_potential_expected_kwh_m2_day": expected_kwh.astype(float),
            "solar_potential_scenario_kwh_m2_day": scenario_kwh.astype(float),
            "solar_potential_kwh_m2_day": final_kwh.astype(float),
            "solar_potential_low_kwh_m2_day": low.astype(float),
            "solar_potential_high_kwh_m2_day": high.astype(float),
            "clear_sky_utilization_ratio": clear_sky_util.astype(float),
            "solar_potential_index": idx.astype(float),
            "solar_potential_class": classify_potential(pd.Series(idx)),
            "is_forecast": merged["is_forecast"].astype(bool),
            "set": np.where(merged["is_forecast"].astype(bool), "forecast", "history"),
            "forecast_lead_month": forecast_lead_month.astype(int),
            "horizon_blend_weight": horizon_blend_weight.astype(float),
            "method": method,
            "uncertainty_method": uncertainty_method,
            "calibration_method": calib.method,
            "model_version": "4.8",
        }
    )
    # Backward-compatible aliases for old column names.
    out["precip_model_mm_day"] = out["precip_model_mm"]
    out["precip_local_mm_day"] = out["precip_local_mm"]

    if "cloud_cover_internet_pct" in merged.columns:
        out["cloud_cover_internet_pct"] = merged["cloud_cover_internet_pct"].astype(float)
    if "wind_speed_internet_kmh" in merged.columns:
        out["wind_speed_internet_kmh"] = merged["wind_speed_internet_kmh"].astype(float)
    if "shortwave_internet_kwh_m2_day" in merged.columns:
        out["shortwave_internet_kwh_m2_day"] = merged["shortwave_internet_kwh_m2_day"].astype(float)
    if "temp_internet_c" in merged.columns:
        out["temperature_internet_c"] = merged["temp_internet_c"].astype(float)
        if "temp_internet_c_harmonized" in merged.columns:
            out["temperature_internet_aligned_c"] = merged["temp_internet_c_harmonized"].astype(float)
            out["temperature_delta_c"] = out["temperature_c"] - out["temperature_internet_aligned_c"]
            out["temperature_delta_raw_c"] = out["temperature_c"] - out["temperature_internet_c"]
        else:
            out["temperature_delta_c"] = out["temperature_c"] - out["temperature_internet_c"]
    if "temp_local" in merged.columns:
        out["temperature_assimilation_adjustment_c"] = out["temperature_model_c"] - out["temperature_local_c"]
    if "temp_assimilation_weight" in merged.columns:
        out["temperature_assimilation_weight"] = merged["temp_assimilation_weight"].astype(float)
    if "humidity_internet_pct" in merged.columns:
        out["humidity_internet_pct"] = merged["humidity_internet_pct"].astype(float)
        if "humidity_internet_pct_harmonized" in merged.columns:
            out["humidity_internet_pct_aligned"] = merged["humidity_internet_pct_harmonized"].astype(float)
            out["humidity_delta_pct"] = out["humidity_model_pct"] - out["humidity_internet_pct_aligned"]
            out["humidity_delta_raw_pct"] = out["humidity_model_pct"] - out["humidity_internet_pct"]
        else:
            out["humidity_delta_pct"] = out["humidity_model_pct"] - out["humidity_internet_pct"]
    if "humidity_assimilation_weight" in merged.columns:
        out["humidity_assimilation_weight"] = merged["humidity_assimilation_weight"].astype(float)
    if "precip_internet_mm_day" in merged.columns:
        out["precip_internet_mm_day"] = merged["precip_internet_mm_day"].astype(float)
    if "precip_internet_mm_month" in merged.columns:
        out["precip_internet_mm_month"] = merged["precip_internet_mm_month"].astype(float)
        if "precip_internet_mm_month_harmonized" in merged.columns:
            out["precip_internet_mm_month_aligned"] = merged["precip_internet_mm_month_harmonized"].astype(float)
            out["precip_delta_mm"] = out["precip_model_mm"] - out["precip_internet_mm_month_aligned"]
            out["precip_delta_raw_mm"] = out["precip_model_mm"] - out["precip_internet_mm_month"]
        else:
            out["precip_delta_mm"] = out["precip_model_mm"] - out["precip_internet_mm_month"]
        out["precip_delta_mm_day"] = out["precip_delta_mm"]
    if "precip_assimilation_weight" in merged.columns:
        out["precip_assimilation_weight"] = merged["precip_assimilation_weight"].astype(float)
    if "pressure_internet" in merged.columns:
        out["pressure_internet"] = merged["pressure_internet"].astype(float)
        if "pressure_internet_harmonized" in merged.columns:
            out["pressure_internet_aligned"] = merged["pressure_internet_harmonized"].astype(float)
            out["pressure_delta"] = out["pressure_model"] - out["pressure_internet_aligned"]
            out["pressure_delta_raw"] = out["pressure_model"] - out["pressure_internet"]
        else:
            out["pressure_delta"] = out["pressure_model"] - out["pressure_internet"]
    if "pressure_assimilation_weight" in merged.columns:
        out["pressure_assimilation_weight"] = merged["pressure_assimilation_weight"].astype(float)

    if "temp_sigma" in merged.columns:
        out["temp_sigma"] = merged["temp_sigma"].astype(float)
        out["humidity_sigma"] = merged["humidity_sigma"].astype(float)
        out["precip_sigma"] = merged["precip_sigma"].astype(float)
        out["pressure_sigma"] = merged["pressure_sigma"].astype(float)

    chart_error = plot_solar_chart(
        out=out,
        solar_obs=solar_obs,
        driver_sensitivity=driver_sensitivity,
        output_chart=args.output_chart,
    )
    chart_status = "ok" if chart_error is None else str(chart_error)
    driver_chart_error = plot_driver_panel_chart(out=out, output_chart=args.output_driver_chart)
    driver_chart_status = "ok" if driver_chart_error is None else str(driver_chart_error)
    if to_bool_text(args.export_separated):
        separated_meta = export_separated_outputs(
            out=out,
            csv_dir=Path(args.separated_output_dir),
            chart_dir=Path(args.separated_chart_dir),
        )
    else:
        separated_meta = {
            "enabled": False,
            "reason": "disabled_by_flag",
            "csv_dir": str(args.separated_output_dir),
            "chart_dir": str(args.separated_chart_dir),
        }

    report: dict[str, Any] = {
        "model_version": "4.8",
        "method_selected": method,
        "mode_requested": mode_req,
        "source_mode_requested": str(args.source_mode),
        "input_source": source_meta,
        "sklearn_available": SKLEARN_OK,
        "rows": int(len(out)),
        "history_rows": int((~out["is_forecast"]).sum()),
        "forecast_rows": int(out["is_forecast"].sum()),
        "date_start": str(pd.to_datetime(out["timestamp"]).min().date()),
        "date_end": str(pd.to_datetime(out["timestamp"]).max().date()),
        "latitude": float(args.latitude),
        "longitude": float(args.longitude),
        "elevation_m": float(args.elevation_m),
        "pv_module_type": str(args.pv_module_type),
        "gamma_pdc": float(gamma_pdc),
        "solar_observation_rows": int(len(solar_obs)),
        "solar_overlap_rows": int(len(overlap)),
        "solar_observations_source": solar_obs_meta,
        "solar_calibration": {
            "enabled": bool(calib.enabled),
            "method": calib.method,
            "overlap_points": int(calib.points),
            "slope": float(calib.slope),
            "intercept": float(calib.intercept),
            "mae": float(calib.mae) if np.isfinite(calib.mae) else None,
            "r2": float(calib.r2) if np.isfinite(calib.r2) else None,
        },
        "ml": ml_meta,
        "uncertainty": {
            "method": uncertainty_method,
            "mc_samples": int(args.mc_samples),
            "uncertainty_z": float(args.uncertainty_z),
            "forecast_smoothing_alpha": float(args.forecast_smoothing_alpha),
            "scenario_requested": bool(scenario_requested),
            "scenario_ar1_rho": float(args.scenario_ar1_rho),
            "scenario_scale": float(args.scenario_scale),
            "horizon_aware_requested": bool(horizon_requested),
            "horizon_aware_applied": bool(horizon_applied),
            "horizon_aware_reason": str(horizon_reason),
            "min_history_for_horizon_blend": int(args.min_history_for_horizon_blend),
            "horizon_h1_months": int(args.horizon_h1_months),
            "horizon_h2_months": int(args.horizon_h2_months),
            "horizon_mid_blend_max": float(args.horizon_mid_blend_max),
            "horizon_long_blend": float(args.horizon_long_blend),
            "horizon_long_blend_growth_per_year": float(args.horizon_long_blend_growth_per_year),
            "horizon_blend_max": float(args.horizon_blend_max),
            "uncertainty_growth_per_year": float(args.uncertainty_growth_per_year),
            "random_seed": int(args.random_seed),
        },
        "quality_checks": build_quality_checks(out),
        "internet_extras": internet_extra_meta,
        "internet_assimilation": internet_assim_meta,
        "internet_shortwave_assimilation": shortwave_assim_meta,
        "internet_cloudiness_blend": {
            "enabled": bool(cloudiness_blend_weight > 0),
            "blend_weight_applied": float(cloudiness_blend_weight),
            "reason": str(cloudiness_blend_reason),
        },
        "internet_consistency": build_internet_consistency(out),
        "input_diagnostics": input_diagnostics,
        "driver_sensitivity": driver_sensitivity,
        "chart": {
            "path": str(args.output_chart),
            "status": chart_status,
            "driver_path": str(args.output_driver_chart),
            "driver_status": driver_chart_status,
            "matplotlib_available": bool(MPL_OK),
        },
        "separated_outputs": separated_meta,
        "clip_quantiles": {"low": float(args.clip_low_q), "high": float(args.clip_high_q)},
        "literature_basis": [
            "FAO-56 Eq.37 clear-sky envelope Rso=(0.75+2e-5*z)*Ra",
            "Erbs et al. (1982) diffuse-fraction correlation from clearness index",
            "PVWatts V5 module temperature coefficient defaults (gamma_pdc)",
            "Reindl et al. (1990) weather-informed regression approach for daily radiation",
            "Thornton & Running (1999) hybrid meteorological estimation practice",
            "Hyndman et al. forecasting practice: autocorrelated simulation for scenario paths",
            "Lead-time dependent uncertainty growth and climatology blending for long horizons",
            "Open-Meteo climate API extras: cloud_cover_mean, wind_speed_10m_mean, shortwave_radiation_sum, temperature_2m_mean",
            "Forecast temperature assimilation: local+internet blend to reduce repeated-pattern bias",
            "Forecast shortwave assimilation: internet shortwave proxy used to nudge expected solar potential",
            "Multi-variable internet assimilation: humidity/precipitation blending with monthly unit harmonization",
            "Internet variable-scale harmonization: affine mapping on historical overlap before forecast blending",
            "Seasonal bias correction + reliability-weighted internet blending for stable multi-variable assimilation",
            "Quantile mapping correction for distribution alignment before seasonal bias correction",
            "NASA POWER monthly ALLSKY_SFC_SW_DWN satellite reference for sparse-target calibration fallback",
            "Conformal residual interval blending with normal approximation for calibrated ML uncertainty bands",
            "Lead-time decay of internet assimilation weights for long-horizon forecast stability",
            "Lagged and rolling meteorological features for improved monthly hybrid ML correction",
            "Time-series cross-validation weighted ensemble for robust hybrid ML generalization",
            "Holdout-validated post-blend between ML ensemble and physics baseline for robustness",
        ],
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    out.to_parquet(args.output_parquet, index=False)
    args.report_json.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(out)} rows.")
    print(f"Method: {method}")
    print(f"Uncertainty: {uncertainty_method}")
    print(f"Source: {source_meta.get('selected_label', 'unknown')}")
    print(f"CSV: {args.output_csv}")
    print(f"Parquet: {args.output_parquet}")
    print(f"Report: {args.report_json}")
    print(f"Chart: {args.output_chart} ({chart_status})")
    print(f"Driver Chart: {args.output_driver_chart} ({driver_chart_status})")
    if bool(separated_meta.get("enabled", False)):
        print(f"Separated CSV Dir: {separated_meta.get('csv_dir')}")
        print(f"Separated Chart Dir: {separated_meta.get('chart_dir')}")


if __name__ == "__main__":
    main()
