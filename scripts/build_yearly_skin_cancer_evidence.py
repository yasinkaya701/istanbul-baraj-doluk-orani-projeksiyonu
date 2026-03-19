#!/usr/bin/env python3
"""Build yearly climate-health evidence focused on skin-cancer pressure.

What this script does:
1) Reads local historical climate tables from Downloads (temperature, humidity, precipitation).
2) Pulls internet climate series from NASA POWER (solar irradiance + cloud amount + meteo context).
3) Builds one yearly track (no multi-model comparison), then projects 2026-2035 with Theil-Sen trend.
4) Produces a transparent per-10k skin-cancer *proxy scenario* and statistical proof tables.

Important scientific note:
- This output is a population-level scenario tool, not a clinical prediction.
- Numeric per-10k values are proxy estimates anchored to explicit assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import kendalltau, linregress, theilslopes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yearly skin-cancer proxy evidence from local + internet climate data")
    p.add_argument("--downloads-root", type=Path, default=Path("/Users/yasinkaya/Downloads"))
    p.add_argument("--output-dir", type=Path, default=Path("output/yearly_skin_cancer_evidence"))
    p.add_argument("--latitude", type=float, default=41.0082, help="Location latitude (default: Istanbul)")
    p.add_argument("--longitude", type=float, default=28.9784, help="Location longitude (default: Istanbul)")
    p.add_argument("--history-start", type=int, default=1987)
    p.add_argument("--history-end", type=int, default=2025)
    p.add_argument("--projection-end", type=int, default=2035)
    p.add_argument("--baseline-start", type=int, default=1991)
    p.add_argument("--baseline-end", type=int, default=2020)
    p.add_argument(
        "--baseline-population",
        type=float,
        default=85561976.0,
        help="Population anchor for per-10k conversion (default: Turkiye 2022)",
    )
    p.add_argument(
        "--baseline-new-cases",
        type=float,
        default=1783.0,
        help="Annual baseline new melanoma cases (default: GLOBOCAN 2022 anchor)",
    )
    p.add_argument("--trend-fit-start", type=int, default=1995, help="Start year for trend-fit window")
    p.add_argument(
        "--station-table1",
        type=Path,
        default=None,
        help="Optional CR800 Table1 path for latest station temp/humidity enrichment",
    )
    p.add_argument(
        "--use-station-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use latest automatic station annual means (if available) to refresh local temp/humidity years",
    )
    p.add_argument(
        "--station-min-records-per-year",
        type=int,
        default=30000,
        help="Minimum 10-min records required to accept a station year as sufficiently complete",
    )
    p.add_argument(
        "--projection-fit-window-years",
        type=int,
        default=20,
        help="Use only the most recent N historical years when fitting projection trends",
    )
    p.add_argument(
        "--projection-damping",
        type=float,
        default=0.92,
        help="Damping factor for future yearly increments (1.0=no damping, lower=flatter tails)",
    )
    p.add_argument(
        "--uv-lag-years",
        type=int,
        default=3,
        help="Rolling-year lag window for effective UV delta before risk mapping",
    )
    p.add_argument(
        "--projected-case-growth-cap-pct",
        type=float,
        default=2.5,
        help="Maximum allowed year-over-year growth (%) in projected per-10k cases",
    )
    p.add_argument(
        "--projected-case-growth-floor-pct",
        type=float,
        default=-0.5,
        help="Minimum allowed year-over-year growth (%) in projected per-10k cases",
    )
    p.add_argument(
        "--align-cases-with-solar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force projected case dynamics to stay directionally parallel with solar dynamics",
    )
    p.add_argument(
        "--solar-parallel-weight",
        type=float,
        default=0.70,
        help="Blend weight of solar YoY into case YoY (0..1)",
    )
    p.add_argument(
        "--solar-parallel-eps-pct",
        type=float,
        default=0.05,
        help="Ignore very small solar YoY moves below this threshold (%) for sign enforcement",
    )
    p.add_argument(
        "--uv-parallel-weight",
        type=float,
        default=0.20,
        help="Blend weight of effective-UV YoY into case YoY (0..1)",
    )
    p.add_argument(
        "--case-growth-inertia",
        type=float,
        default=0.35,
        help="Inertia for projected case YoY (0=no memory, 0.95=very smooth)",
    )
    p.add_argument(
        "--growth-guardrail-decay",
        type=float,
        default=0.04,
        help="Per-year decay of growth cap/floor magnitude in projection horizon",
    )
    p.add_argument(
        "--projection-noise-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add controlled stochastic noise to projected climate drivers",
    )
    p.add_argument(
        "--projection-noise-strength",
        type=float,
        default=0.35,
        help="Noise strength multiplier over robust historical residual scale",
    )
    p.add_argument(
        "--projection-noise-ar1",
        type=float,
        default=0.55,
        help="AR(1) persistence for projection noise (0..0.98)",
    )
    p.add_argument(
        "--projection-noise-seed",
        type=int,
        default=42,
        help="Base random seed for projection noise",
    )
    p.add_argument(
        "--projection-noise-min-frac",
        type=float,
        default=0.003,
        help="Minimum noise sigma as fraction of anchor level",
    )
    p.add_argument(
        "--projection-noise-max-frac",
        type=float,
        default=0.040,
        help="Maximum noise sigma as fraction of anchor level",
    )
    return p.parse_args()


@dataclass
class FitMeta:
    intercept: float
    slope: float
    r2: float
    n_overlap: int


@dataclass
class TrendMeta:
    intercept: float
    slope: float
    slope_low: float
    slope_high: float
    fit_start: int
    fit_end: int
    n_fit: int


def _to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _year_from_col(col: object) -> int | None:
    if isinstance(col, (int, np.integer)):
        y = int(col)
        if 1800 <= y <= 2300:
            return y
    if isinstance(col, float) and np.isfinite(col):
        y = int(round(col))
        if abs(col - y) < 1e-6 and 1800 <= y <= 2300:
            return y
    m = re.search(r"(18|19|20|21)\d{2}", str(col))
    if not m:
        return None
    y = int(m.group(0))
    if 1800 <= y <= 2300:
        return y
    return None


def extract_yearly_transposed_excel(
    path: Path,
    sheet_name: str,
    header: int,
    agg: str,
    value_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing local file: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    year_cols: dict[int, object] = {}
    for c in df.columns:
        y = _year_from_col(c)
        if y is None:
            continue
        year_cols[y] = c

    if not year_cols:
        raise SystemExit(f"No year columns found in {path} [{sheet_name}]")

    rows: list[dict[str, float]] = []
    for y in sorted(year_cols):
        col = year_cols[y]
        vals = _to_numeric(df[col])
        if agg == "mean":
            v = float(vals.mean())
        elif agg == "sum":
            v = float(vals.sum(min_count=1))
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")
        if np.isfinite(v):
            rows.append({"year": int(y), value_name: v})

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def load_station_annual(table1_path: Path, min_records_per_year: int) -> pd.DataFrame:
    cols = ["year", "temp_station_c", "humidity_station_pct", "station_records_temp", "station_records_humidity"]
    if not table1_path.exists():
        return pd.DataFrame(columns=cols)

    raw = pd.read_csv(table1_path, skiprows=[0, 2, 3], dtype=str, low_memory=False)
    if "TIMESTAMP" not in raw.columns:
        return pd.DataFrame(columns=cols)

    raw["timestamp"] = pd.to_datetime(raw["TIMESTAMP"], errors="coerce")
    raw["year"] = raw["timestamp"].dt.year
    raw["temp_station_c"] = _to_numeric(raw.get("AirTCee181_Avg", pd.Series(np.nan, index=raw.index)))
    raw["humidity_station_pct"] = _to_numeric(raw.get("RHee181_Avg", pd.Series(np.nan, index=raw.index)))

    raw = raw.dropna(subset=["year"]).copy()
    raw["year"] = raw["year"].astype(int)

    grp = (
        raw.groupby("year", as_index=False)
        .agg(
            temp_station_c=("temp_station_c", "mean"),
            humidity_station_pct=("humidity_station_pct", "mean"),
            station_records_temp=("temp_station_c", lambda s: int(s.notna().sum())),
            station_records_humidity=("humidity_station_pct", lambda s: int(s.notna().sum())),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    # Keep only sufficiently complete years.
    m = (grp["station_records_temp"] >= int(min_records_per_year)) & (
        grp["station_records_humidity"] >= int(min_records_per_year)
    )
    return grp[m].reset_index(drop=True)


def fetch_nasa_power_monthly(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,CLOUD_AMT,T2M,RH2M,PRECTOTCORR",
        "community": "SB",
        "latitude": float(lat),
        "longitude": float(lon),
        "start": str(int(start_year)),
        "end": str(int(end_year)),
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=120)
    if resp.status_code != 200:
        raise SystemExit(f"NASA POWER request failed: HTTP {resp.status_code} -> {resp.text[:400]}")
    payload = resp.json()
    pdata = payload.get("properties", {}).get("parameter", {})
    if not pdata:
        raise SystemExit("NASA POWER payload missing parameter block")

    out_rows: list[dict[str, Any]] = []
    for p_name, mapping in pdata.items():
        if not isinstance(mapping, dict):
            continue
        for ym, raw_val in mapping.items():
            sval = float(raw_val) if raw_val is not None else np.nan
            if not np.isfinite(sval) or sval <= -998.0:
                continue
            try:
                year = int(str(ym)[:4])
                month = int(str(ym)[4:6])
                ts = pd.Timestamp(year=year, month=month, day=1)
            except Exception:
                continue
            out_rows.append({"timestamp": ts, "parameter": p_name, "value": float(sval)})

    if not out_rows:
        raise SystemExit("NASA POWER returned no usable rows")

    long_df = pd.DataFrame(out_rows)
    wide = long_df.pivot_table(index="timestamp", columns="parameter", values="value", aggfunc="mean").reset_index()
    wide = wide.sort_values("timestamp").reset_index(drop=True)
    return wide


def nasa_monthly_to_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    m["year"] = pd.to_datetime(m["timestamp"]).dt.year
    m["month"] = pd.to_datetime(m["timestamp"]).dt.month
    m["days_in_month"] = pd.to_datetime(m["timestamp"]).dt.days_in_month

    # NASA monthly shortwave unit is W/m2. Convert to kWh/m2/day.
    if "ALLSKY_SFC_SW_DWN" not in m.columns:
        raise SystemExit("NASA monthly table missing ALLSKY_SFC_SW_DWN")
    if "CLOUD_AMT" not in m.columns:
        raise SystemExit("NASA monthly table missing CLOUD_AMT")

    m["solar_kwh_m2_day"] = _to_numeric(m["ALLSKY_SFC_SW_DWN"]) * 24.0 / 1000.0
    m["cloud_pct"] = _to_numeric(m["CLOUD_AMT"])
    m["temp_nasa_c"] = _to_numeric(m.get("T2M", pd.Series(np.nan, index=m.index)))
    m["humidity_nasa_pct"] = _to_numeric(m.get("RH2M", pd.Series(np.nan, index=m.index)))
    m["precip_nasa_mm_month"] = _to_numeric(m.get("PRECTOTCORR", pd.Series(np.nan, index=m.index))) * m["days_in_month"]

    annual = (
        m.groupby("year", as_index=False)
        .agg(
            solar_kwh_m2_day=("solar_kwh_m2_day", "mean"),
            cloud_pct=("cloud_pct", "mean"),
            temp_nasa_c=("temp_nasa_c", "mean"),
            humidity_nasa_pct=("humidity_nasa_pct", "mean"),
            precip_nasa_mm=("precip_nasa_mm_month", "sum"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    return annual


def fit_local_from_nasa(local: pd.Series, nasa: pd.Series) -> FitMeta:
    ov = local.dropna().index.intersection(nasa.dropna().index)
    if len(ov) < 8:
        return FitMeta(intercept=0.0, slope=1.0, r2=0.0, n_overlap=int(len(ov)))
    x = nasa.loc[ov].astype(float)
    y = local.loc[ov].astype(float)
    lr = linregress(x, y)
    r2 = float(lr.rvalue ** 2) if np.isfinite(lr.rvalue) else 0.0
    if not np.isfinite(lr.intercept) or not np.isfinite(lr.slope):
        return FitMeta(intercept=0.0, slope=1.0, r2=0.0, n_overlap=int(len(ov)))
    return FitMeta(intercept=float(lr.intercept), slope=float(lr.slope), r2=r2, n_overlap=int(len(ov)))


def fill_missing_with_nasa_mapping(
    local: pd.Series,
    nasa: pd.Series,
    history_end: int,
) -> tuple[pd.Series, FitMeta]:
    out = local.copy()
    fit = fit_local_from_nasa(local=local, nasa=nasa)
    missing_hist = out.index[(out.isna()) & (out.index <= int(history_end))]
    if len(missing_hist):
        out.loc[missing_hist] = fit.intercept + fit.slope * nasa.loc[missing_hist]
    return out, fit


def project_theilsen(
    series: pd.Series,
    history_end: int,
    projection_end: int,
    fit_start: int,
    damping: float = 1.0,
) -> tuple[pd.Series, TrendMeta]:
    out = series.copy()
    fit_data = out[(out.index >= int(fit_start)) & (out.index <= int(history_end))].dropna()
    if len(fit_data) < 8:
        raise SystemExit(f"Not enough data for Theil-Sen projection: n={len(fit_data)}")

    x = fit_data.index.astype(float).to_numpy()
    y = fit_data.astype(float).to_numpy()
    slope, intercept, slope_low, slope_high = theilslopes(y, x, 0.95)

    # Keep continuity at history_end to avoid an artificial jump at projection start.
    if int(history_end) not in out.index or not np.isfinite(float(out.loc[int(history_end)])):
        raise SystemExit(f"History end year {history_end} missing for projection anchor")
    damp = float(damping)
    if not np.isfinite(damp) or damp <= 0.0 or damp > 1.0:
        raise SystemExit(f"Invalid damping value: {damping}. Must be in (0, 1].")
    anchor_value = float(out.loc[int(history_end)])
    running = anchor_value
    for i, yr in enumerate(range(int(history_end) + 1, int(projection_end) + 1), start=1):
        increment = float(slope * (damp ** (i - 1)))
        running = float(running + increment)
        out.loc[yr] = running

    meta = TrendMeta(
        intercept=float(intercept),
        slope=float(slope),
        slope_low=float(slope_low),
        slope_high=float(slope_high),
        fit_start=int(fit_data.index.min()),
        fit_end=int(fit_data.index.max()),
        n_fit=int(len(fit_data)),
    )
    return out, meta


def clamp_physical(s: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    out = s.copy()
    if lower is not None:
        out = out.clip(lower=lower)
    if upper is not None:
        out = out.clip(upper=upper)
    return out


def compute_projection_fit_start(history_start: int, history_end: int, trend_fit_start: int, fit_window_years: int) -> int:
    window = max(8, int(fit_window_years))
    recent_start = int(history_end) - window + 1
    fit_start = max(int(history_start), int(trend_fit_start), int(recent_start))
    return int(fit_start)


def _name_seed_offset(name: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(str(name))) % 1000003)


def add_projection_noise(
    series: pd.Series,
    history_end: int,
    projection_end: int,
    fit_start: int,
    enable: bool,
    strength: float,
    ar1: float,
    seed: int,
    name: str,
    min_frac: float,
    max_frac: float,
) -> tuple[pd.Series, dict[str, float]]:
    out = series.copy()
    meta = {
        "enabled": bool(enable),
        "sigma_used": 0.0,
        "ar1": float(ar1),
        "strength": float(strength),
        "n_projection_years": max(0, int(projection_end) - int(history_end)),
    }
    if not bool(enable):
        return out, meta
    if int(projection_end) <= int(history_end):
        return out, meta

    fit_data = out[(out.index >= int(fit_start)) & (out.index <= int(history_end))].dropna()
    if len(fit_data) < 8:
        return out, meta

    x = fit_data.index.astype(float).to_numpy()
    y = fit_data.astype(float).to_numpy()
    slope, intercept, _, _ = theilslopes(y, x, 0.95)
    fitted = intercept + slope * x
    resid = y - fitted
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    robust_sigma = float(1.4826 * mad)
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = 0.0

    anchor_candidates = [
        abs(float(out.loc[int(history_end)])) if int(history_end) in out.index else np.nan,
        abs(float(fit_data.mean())),
    ]
    anchor = max([v for v in anchor_candidates if np.isfinite(v)] + [1e-6])

    min_sigma = float(max(0.0, min_frac) * anchor)
    max_sigma = float(max(min_sigma, max_frac * anchor))
    sigma = float(np.clip(robust_sigma * float(strength), min_sigma, max_sigma))
    if not np.isfinite(sigma) or sigma <= 0:
        return out, meta

    phi = float(np.clip(ar1, 0.0, 0.98))
    innovation_scale = float(np.sqrt(max(1e-9, 1.0 - phi ** 2)))
    rng = np.random.default_rng(int(seed) + _name_seed_offset(name))
    shock = 0.0
    for yr in range(int(history_end) + 1, int(projection_end) + 1):
        shock = float(phi * shock + innovation_scale * rng.normal(0.0, 1.0))
        out.loc[yr] = float(out.loc[yr]) + float(sigma * shock)

    meta["sigma_used"] = float(sigma)
    meta["anchor_level"] = float(anchor)
    meta["robust_sigma_hist"] = float(robust_sigma)
    return out, meta


def apply_projected_growth_guardrail(
    annual_idx: pd.DataFrame,
    history_end: int,
    projection_end: int,
    growth_floor_pct: float,
    growth_cap_pct: float,
    align_with_solar: bool,
    solar_parallel_weight: float,
    solar_parallel_eps_pct: float,
    uv_parallel_weight: float,
    case_growth_inertia: float,
    growth_guardrail_decay: float,
) -> pd.DataFrame:
    out = annual_idx.copy()
    floor = float(growth_floor_pct) / 100.0
    cap = float(growth_cap_pct) / 100.0
    if floor > cap:
        floor, cap = cap, floor
    ws = float(np.clip(float(solar_parallel_weight), 0.0, 1.0))
    wu = float(np.clip(float(uv_parallel_weight), 0.0, 1.0))
    if ws + wu > 1.0:
        s = ws + wu
        ws = ws / s
        wu = wu / s
    wr = float(max(0.0, 1.0 - ws - wu))
    inertia = float(np.clip(float(case_growth_inertia), 0.0, 0.95))
    decay = float(np.clip(float(growth_guardrail_decay), 0.0, 0.20))
    eps = abs(float(solar_parallel_eps_pct)) / 100.0

    # Use raw projected growth rates as reference so the guardrail does not
    # recursively force every later year to the cap.
    raw_mid = annual_idx["cases_per10k_mid"].astype(float).copy()
    raw_low = annual_idx["cases_per10k_low"].astype(float).copy()
    raw_high = annual_idx["cases_per10k_high"].astype(float).copy()
    raw_solar = annual_idx["solar_kwh_m2_day"].astype(float).copy()
    raw_uv = annual_idx["effective_uv_kwh_m2_day"].astype(float).copy()
    prev_adj_growth: float | None = None

    for yr in range(int(history_end) + 1, int(projection_end) + 1):
        if (yr - 1) not in out.index or yr not in out.index or (yr - 1) not in raw_mid.index or yr not in raw_mid.index:
            continue

        prev_mid = float(out.loc[yr - 1, "cases_per10k_mid"])
        raw_prev_mid = float(raw_mid.loc[yr - 1])
        raw_cur_mid = float(raw_mid.loc[yr])
        if (
            not np.isfinite(prev_mid)
            or not np.isfinite(raw_prev_mid)
            or not np.isfinite(raw_cur_mid)
            or prev_mid <= 0
            or raw_prev_mid <= 0
        ):
            continue

        growth_raw = float(raw_cur_mid / raw_prev_mid - 1.0)
        growth = float(growth_raw)
        solar_growth = float("nan")
        uv_growth = float("nan")

        if (yr - 1) in raw_solar.index and yr in raw_solar.index:
            s_prev = float(raw_solar.loc[yr - 1])
            s_cur = float(raw_solar.loc[yr])
            if np.isfinite(s_prev) and np.isfinite(s_cur) and s_prev > 0:
                solar_growth = float(s_cur / s_prev - 1.0)
        if (yr - 1) in raw_uv.index and yr in raw_uv.index:
            u_prev = float(raw_uv.loc[yr - 1])
            u_cur = float(raw_uv.loc[yr])
            if np.isfinite(u_prev) and np.isfinite(u_cur) and u_prev > 0:
                uv_growth = float(u_cur / u_prev - 1.0)

        target = float(wr * growth_raw)
        if np.isfinite(solar_growth):
            target += float(ws * solar_growth)
        else:
            target += float(ws * growth_raw)
        if np.isfinite(uv_growth):
            target += float(wu * uv_growth)
        else:
            target += float(wu * growth_raw)

        if prev_adj_growth is not None:
            growth = float(inertia * prev_adj_growth + (1.0 - inertia) * target)
        else:
            growth = float(target)

        # Keep direction parallel with solar when solar move is meaningful.
        if bool(align_with_solar) and np.isfinite(solar_growth):
            if solar_growth > eps and growth < 0:
                growth = max(0.0, 0.25 * solar_growth)
            elif solar_growth < -eps and growth > 0:
                growth = min(0.0, 0.25 * solar_growth)

        h = max(1, int(yr - int(history_end)))
        shrink = float(max(0.65, 1.0 - decay * float(h - 1)))
        cap_h = float(cap * shrink)
        floor_h = float(floor * shrink)
        if floor_h > cap_h:
            floor_h, cap_h = cap_h, floor_h

        clipped = float(np.clip(growth, floor_h, cap_h))
        prev_adj_growth = float(clipped)

        adj_mid = float(prev_mid * (1.0 + clipped))
        scale = float(adj_mid / raw_cur_mid) if raw_cur_mid != 0 else 1.0
        out.loc[yr, "cases_per10k_low"] = float(raw_low.loc[yr]) * scale
        out.loc[yr, "cases_per10k_mid"] = float(raw_mid.loc[yr]) * scale
        out.loc[yr, "cases_per10k_high"] = float(raw_high.loc[yr]) * scale

    return out


def trend_proof(series: pd.Series, label: str, unit: str, start: int, end: int) -> dict[str, Any]:
    s = series[(series.index >= int(start)) & (series.index <= int(end))].dropna()
    if len(s) < 8:
        return {
            "variable": label,
            "unit": unit,
            "n": int(len(s)),
            "slope_per_year": np.nan,
            "slope_per_decade": np.nan,
            "p_value": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
            "theil_sen_slope": np.nan,
            "theil_sen_low": np.nan,
            "theil_sen_high": np.nan,
            "direction": "insufficient_data",
        }

    x = s.index.astype(float).to_numpy()
    y = s.astype(float).to_numpy()
    lr = linregress(x, y)
    tau, tau_p = kendalltau(x, y)
    ts_slope, _, ts_low, ts_high = theilslopes(y, x, 0.95)

    direction = "flat"
    if np.isfinite(lr.slope):
        if lr.slope > 0:
            direction = "up"
        elif lr.slope < 0:
            direction = "down"

    return {
        "variable": label,
        "unit": unit,
        "n": int(len(s)),
        "slope_per_year": float(lr.slope),
        "slope_per_decade": float(lr.slope * 10.0),
        "p_value": float(lr.pvalue),
        "kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
        "kendall_p": float(tau_p) if np.isfinite(tau_p) else np.nan,
        "theil_sen_slope": float(ts_slope),
        "theil_sen_low": float(ts_low),
        "theil_sen_high": float(ts_high),
        "direction": direction,
    }


def plot_climate_drivers(df: pd.DataFrame, history_end: int, out_png: Path) -> None:
    h = df.copy()
    h = h.sort_values("year")
    years = h["year"].to_numpy(dtype=int)
    is_proj = h["is_projected"].to_numpy(dtype=bool)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1) Solar + cloud
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(years, h["solar_kwh_m2_day"], color="#E69F00", linewidth=2, label="Gunes (kWh/m2/gun)")
    ax2.plot(years, h["cloud_pct"], color="#56B4E9", linewidth=2, label="Bulut (%)")
    ax.axvline(history_end + 0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Yillik Gunes ve Bulutluluk")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Gunes")
    ax2.set_ylabel("Bulut")
    ax.grid(alpha=0.25)

    # 2) Precip + humidity
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(years, h["precip_mm"], color="#009E73", linewidth=2, label="Yagis (mm/yil)")
    ax2.plot(years, h["humidity_pct"], color="#0072B2", linewidth=2, label="Nem (%)")
    ax.axvline(history_end + 0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Yagis ve Nem")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Yagis")
    ax2.set_ylabel("Nem")
    ax.grid(alpha=0.25)

    # 3) Temperature
    ax = axes[1, 0]
    ax.plot(years, h["temp_c"], color="#D55E00", linewidth=2)
    ax.axvline(history_end + 0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Yillik Ortalama Sicaklik")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Sicaklik (C)")
    ax.grid(alpha=0.25)

    # 4) Effective UV proxy
    ax = axes[1, 1]
    ax.plot(years, h["effective_uv_kwh_m2_day"], color="#CC79A7", linewidth=2, label="Etkili UV proxy")
    hist = h[~h["is_projected"]]
    if len(hist) >= 8:
        x = hist["year"].to_numpy(dtype=float)
        y = hist["effective_uv_kwh_m2_day"].to_numpy(dtype=float)
        lr = linregress(x, y)
        trend = lr.intercept + lr.slope * x
        ax.plot(hist["year"], trend, color="#111111", linestyle="--", linewidth=1.5, label="Trend (OLS)")
    ax.axvline(history_end + 0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Etkili UV Proxy (Gunes x (1-Bulut))")
    ax.set_xlabel("Yil")
    ax.set_ylabel("kWh/m2/gun")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Shade projected area on all panels
    proj_years = years[is_proj]
    if len(proj_years):
        x0 = int(proj_years.min()) - 0.5
        x1 = int(proj_years.max()) + 0.5
        for a in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
            a.axvspan(x0, x1, color="#f3f3f3", alpha=0.6, zorder=0)

    fig.suptitle("Yillara Gore Iklim Suruculeri (Yerel + Internet)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_skin_cases(df: pd.DataFrame, history_end: int, out_png: Path) -> None:
    d = df.sort_values("year").copy()
    years = d["year"].to_numpy(dtype=int)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(13.5, 6.0))

    ax.fill_between(
        years,
        d["cases_per10k_low"].to_numpy(dtype=float),
        d["cases_per10k_high"].to_numpy(dtype=float),
        color="#ffd9b3",
        alpha=0.5,
        label="Belirsizlik bandi (dusuk-yuksek)",
    )
    ax.plot(years, d["cases_per10k_mid"], color="#d95f02", linewidth=2.5, label="Merkez senaryo")
    ax.axvline(history_end + 0.5, color="#555555", linestyle="--", linewidth=1.2)

    ax.set_title("Cilt Kanseri Proxy Senaryosu (10.000 kiside, yillik)")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Vaka / 10.000")
    ax.grid(alpha=0.25)

    y_2025 = d.loc[d["year"] == int(history_end), "cases_per10k_mid"]
    y_2035 = d.loc[d["year"] == int(d["year"].max()), "cases_per10k_mid"]
    if len(y_2025) and len(y_2035):
        ax.text(
            0.02,
            0.96,
            f"{history_end}: {float(y_2025.iloc[0]):.4f}/10k\\n{int(d['year'].max())}: {float(y_2035.iloc[0]):.4f}/10k",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": "#b0b0b0"},
        )

    ax.legend(loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_evidence_dashboard(trend_df: pd.DataFrame, corr_df: pd.DataFrame, out_png: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # Left: trend significance bars
    t = trend_df.copy()
    t = t[np.isfinite(t["p_value"])].copy()
    t["neglog10_p"] = -np.log10(np.clip(t["p_value"].to_numpy(dtype=float), 1e-12, 1.0))
    colors = ["#2ca02c" if d == "up" else "#1f77b4" if d == "down" else "#7f7f7f" for d in t["direction"]]
    axes[0].barh(t["variable"], t["neglog10_p"], color=colors)
    axes[0].axvline(-math.log10(0.05), color="#444444", linestyle="--", linewidth=1)
    axes[0].set_title("Trend Kaniti (-log10 p)")
    axes[0].set_xlabel("Guclu kanit  ->")
    axes[0].grid(axis="x", alpha=0.25)

    # Right: correlation heatmap
    c = corr_df.to_numpy(dtype=float)
    im = axes[1].imshow(c, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    labels = list(corr_df.columns)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[1].set_title("Surucu Degisken Korelasyonlari (Tarihsel)")
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            v = c[i, j]
            txt = f"{v:.2f}" if np.isfinite(v) else "NA"
            axes[1].text(j, i, txt, ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Istatistiksel Kanit Panosu", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_markdown_report(
    out_md: Path,
    annual: pd.DataFrame,
    trend_df: pd.DataFrame,
    literature_consistency: str,
    assumptions: dict[str, Any],
    history_end: int,
    projection_end: int,
) -> None:
    d = annual.sort_values("year").copy()
    h = d[d["is_projected"] == False].copy()
    p = d[d["is_projected"] == True].copy()

    y0 = int(h["year"].min()) if not h.empty else int(d["year"].min())
    y1 = int(history_end)

    def val(year: int, col: str) -> float:
        x = d.loc[d["year"] == int(year), col]
        return float(x.iloc[0]) if len(x) else float("nan")

    eff_start = val(y0, "effective_uv_kwh_m2_day")
    eff_end = val(y1, "effective_uv_kwh_m2_day")
    c25 = val(history_end, "cases_per10k_mid")
    c35_low = val(projection_end, "cases_per10k_low")
    c35_mid = val(projection_end, "cases_per10k_mid")
    c35_high = val(projection_end, "cases_per10k_high")

    eff_delta_pct = np.nan
    if np.isfinite(eff_start) and abs(eff_start) > 1e-9 and np.isfinite(eff_end):
        eff_delta_pct = (eff_end - eff_start) / eff_start * 100.0

    tr_eff = trend_df[trend_df["variable"] == "effective_uv_kwh_m2_day"]
    tr_case = trend_df[trend_df["variable"] == "cases_per10k_mid"]
    eff_p = float(tr_eff["p_value"].iloc[0]) if len(tr_eff) else np.nan
    case_p = float(tr_case["p_value"].iloc[0]) if len(tr_case) else np.nan

    lines = [
        "# Yillik Cilt Kanseri Etki Analizi (Tek-Yontem, Kanitli)",
        "",
        "## Kisa Sonuc",
        f"- Incelenen donem: {y0}-{y1} (tarihsel), {y1+1}-{projection_end} (trend projeksiyonu)",
        f"- Etkili UV proxy degisimi ({y0}->{y1}): {eff_delta_pct:+.2f}%" if np.isfinite(eff_delta_pct) else "- Etkili UV proxy degisimi hesaplanamadi.",
        f"- 10.000 kiside merkez senaryo: {history_end} icin {c25:.4f}, {projection_end} icin {c35_mid:.4f}",
        f"- {projection_end} belirsizlik araligi: [{c35_low:.4f}, {c35_high:.4f}] / 10.000",
        f"- Literatur uyum kontrolu: {literature_consistency}",
        "",
        "## Kanit Ozeti",
        f"- Etkili UV trend p-degeri: {eff_p:.4g}" if np.isfinite(eff_p) else "- Etkili UV trend p-degeri hesaplanamadi.",
        f"- Vaka/10k trend p-degeri: {case_p:.4g}" if np.isfinite(case_p) else "- Vaka/10k trend p-degeri hesaplanamadi.",
        "- Trend testleri: OLS + Kendall tau + Theil-Sen egimi birlikte raporlandi.",
        "",
        "## Yontem (Sade)",
        "- Yerel veri (Downloads): sicaklik, nem, yagis yillik serileri cikartildi.",
        "- Internet veri (NASA POWER): gunes (ALLSKY_SFC_SW_DWN), bulut (CLOUD_AMT), destek meteo serileri alindi.",
        "- Etkili UV proxy = gunes_kwh_m2_gun x (1 - bulut/100).",
        "- Cilt kanseri per-10k senaryosu, 2022 baseline vaka/populasyon capasi ve WHO ozon-UV risk notlariyla kuruldu.",
        "- 2026-2035 icin tek yaklasim: Theil-Sen trend uzatimi (model yarisi yok).",
        f"- Gercekcilik katmani: son {int(assumptions.get('projection_fit_window_years', 20))} yildan fit + damping={float(assumptions.get('projection_damping', 1.0)):.2f}.",
        f"- Gercekcilik katmani: UV etkisinde {int(assumptions.get('uv_lag_years', 1))} yillik gecikmeli ortalama kullanildi.",
        (
            f"- Gercekcilik katmani: projeksiyona kontrollu gurultu eklendi "
            f"(enable={bool(assumptions.get('projection_noise_enable', False))}, "
            f"strength={float(assumptions.get('projection_noise_strength', 0.0)):.2f}, "
            f"AR1={float(assumptions.get('projection_noise_ar1', 0.0)):.2f})."
        ),
        (
            f"- Gercekcilik katmani: projeksiyon yillik vaka artis siniri "
            f"[{float(assumptions.get('projected_case_growth_floor_pct', -99.0)):.2f}%, "
            f"{float(assumptions.get('projected_case_growth_cap_pct', 99.0)):.2f}%]."
        ),
        (
            f"- Gercekcilik katmani: vaka-gunes paralellik ayari "
            f"(enable={bool(assumptions.get('align_cases_with_solar', False))}, "
            f"weight={float(assumptions.get('solar_parallel_weight', 0.0)):.2f}, "
            f"eps={float(assumptions.get('solar_parallel_eps_pct', 0.0)):.3f}%)."
        ),
        (
            f"- Gercekcilik katmani: UV paralellik + atalet ayari "
            f"(uv_weight={float(assumptions.get('uv_parallel_weight', 0.0)):.2f}, "
            f"inertia={float(assumptions.get('case_growth_inertia', 0.0)):.2f}, "
            f"guardrail_decay={float(assumptions.get('growth_guardrail_decay', 0.0)):.3f})."
        ),
        "",
        "## Sinirlar",
        "- Bu cikti klinik tani araci degildir; nufus duzeyi risk-baski senaryosudur.",
        "- UV-B dogrudan olculmedi; etkili UV proxy kullanildi.",
        "- Per-10k sayilari nedensel kesin tahmin degil, acik varsayimli senaryodur.",
        "",
        "## Varsayimlar",
        f"- baseline_population = {assumptions['baseline_population']}",
        f"- baseline_new_cases = {assumptions['baseline_new_cases']}",
        f"- baseline_cases_per10k = {assumptions['baseline_cases_per10k']:.6f}",
        f"- rr_beta_low_per_pct = {assumptions['rr_beta_low_per_pct']:.8f}",
        f"- rr_beta_mid_per_pct = {assumptions['rr_beta_mid_per_pct']:.8f}",
        f"- rr_beta_high_per_pct = {assumptions['rr_beta_high_per_pct']:.8f}",
        "",
        "## Kaynaklar",
        "- WHO Q&A (UV ve cilt kanseri): https://www.who.int/news-room/questions-and-answers/item/radiation-ultraviolet-(uv)-radiation-and-skin-cancer",
        "- WHO UV fact sheet: https://www.who.int/news-room/fact-sheets/detail/ultraviolet-radiation",
        "- WHO-ILO (mesleki gunes maruziyeti): https://www.who.int/news/item/08-11-2023-working-under-the-sun-causes-1-in-3-deaths-from-non-melanoma-skin-cancer--say-who-and-ilo",
        "- IARC 2025 UV bulgusu: https://www.iarc.who.int/news-events/world-cancer-day-2025-illuminating-the-role-of-uv-radiation-in-skin-cancer-risk/",
        "- Occupational UV meta-analizi (PubMed): https://pubmed.ncbi.nlm.nih.gov/21054335/",
        "- NASA POWER API: https://power.larc.nasa.gov/",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    out_root = args.output_dir.resolve()
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Local historical tables (Downloads)
    temp_path = args.downloads_root / "Veriler_H" / "Sçcaklçk" / "Uzunyçl-Max-Min-Ort-orj.xlsx"
    hum_path = args.downloads_root / "Veriler_H" / "Nem" / "1911-2022-Nem.xlsx"
    precip_path = args.downloads_root / "Veriler_H" / "Yaßçü" / "1911-2023.xlsx"
    station_table1_path = (
        args.station_table1
        if args.station_table1 is not None
        else args.downloads_root / "Veriler_H" / "Otomatik òstasyon" / "CR800Series_Table1.dat"
    )

    temp_local = extract_yearly_transposed_excel(
        path=temp_path,
        sheet_name="Ort",
        header=0,
        agg="mean",
        value_name="temp_local_c",
    )
    hum_local = extract_yearly_transposed_excel(
        path=hum_path,
        sheet_name="Nem 1911-",
        header=1,
        agg="mean",
        value_name="humidity_local_pct",
    )
    precip_local = extract_yearly_transposed_excel(
        path=precip_path,
        sheet_name="Günlük",
        header=0,
        agg="sum",
        value_name="precip_local_mm",
    )

    station_annual = load_station_annual(
        table1_path=station_table1_path,
        min_records_per_year=int(args.station_min_records_per_year),
    )
    if bool(args.use_station_latest) and not station_annual.empty:
        st_temp = station_annual[["year", "temp_station_c"]].rename(columns={"temp_station_c": "temp_local_c"})
        st_hum = station_annual[["year", "humidity_station_pct"]].rename(
            columns={"humidity_station_pct": "humidity_local_pct"}
        )
        temp_local = (
            pd.concat([temp_local, st_temp], ignore_index=True)
            .sort_values("year")
            .drop_duplicates(subset=["year"], keep="last")
            .reset_index(drop=True)
        )
        hum_local = (
            pd.concat([hum_local, st_hum], ignore_index=True)
            .sort_values("year")
            .drop_duplicates(subset=["year"], keep="last")
            .reset_index(drop=True)
        )

    # 2) Internet climate from NASA POWER
    nasa_monthly = fetch_nasa_power_monthly(
        lat=float(args.latitude),
        lon=float(args.longitude),
        start_year=int(args.history_start),
        end_year=int(args.history_end),
    )
    nasa_annual = nasa_monthly_to_annual(nasa_monthly)

    # 3) Build yearly frame
    years = np.arange(int(args.history_start), int(args.projection_end) + 1)
    annual = pd.DataFrame({"year": years})
    annual = annual.merge(nasa_annual, on="year", how="left")
    annual = annual.merge(temp_local, on="year", how="left")
    annual = annual.merge(hum_local, on="year", how="left")
    annual = annual.merge(precip_local, on="year", how="left")
    annual = annual.sort_values("year").reset_index(drop=True)
    annual["is_projected"] = annual["year"] > int(args.history_end)

    annual_idx = annual.set_index("year")

    # 3a) Fill missing historical local values with NASA-mapped equivalents
    temp_filled, temp_fit = fill_missing_with_nasa_mapping(
        local=annual_idx["temp_local_c"],
        nasa=annual_idx["temp_nasa_c"],
        history_end=int(args.history_end),
    )
    hum_filled, hum_fit = fill_missing_with_nasa_mapping(
        local=annual_idx["humidity_local_pct"],
        nasa=annual_idx["humidity_nasa_pct"],
        history_end=int(args.history_end),
    )
    precip_filled, precip_fit = fill_missing_with_nasa_mapping(
        local=annual_idx["precip_local_mm"],
        nasa=annual_idx["precip_nasa_mm"],
        history_end=int(args.history_end),
    )

    annual_idx["temp_c"] = temp_filled
    annual_idx["humidity_pct"] = hum_filled
    annual_idx["precip_mm"] = precip_filled

    # 3b) Project all drivers to projection_end with Theil-Sen (single method)
    fit_start = compute_projection_fit_start(
        history_start=int(args.history_start),
        history_end=int(args.history_end),
        trend_fit_start=int(args.trend_fit_start),
        fit_window_years=int(args.projection_fit_window_years),
    )
    solar_proj, solar_meta = project_theilsen(
        series=annual_idx["solar_kwh_m2_day"],
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        damping=float(args.projection_damping),
    )
    cloud_proj, cloud_meta = project_theilsen(
        series=annual_idx["cloud_pct"],
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        damping=float(args.projection_damping),
    )
    temp_proj, temp_meta = project_theilsen(
        series=annual_idx["temp_c"],
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        damping=float(args.projection_damping),
    )
    hum_proj, hum_meta = project_theilsen(
        series=annual_idx["humidity_pct"],
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        damping=float(args.projection_damping),
    )
    precip_proj, precip_meta = project_theilsen(
        series=annual_idx["precip_mm"],
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        damping=float(args.projection_damping),
    )

    solar_proj, solar_noise_meta = add_projection_noise(
        series=solar_proj,
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        enable=bool(args.projection_noise_enable),
        strength=float(args.projection_noise_strength),
        ar1=float(args.projection_noise_ar1),
        seed=int(args.projection_noise_seed),
        name="solar",
        min_frac=float(args.projection_noise_min_frac),
        max_frac=float(args.projection_noise_max_frac),
    )
    cloud_proj, cloud_noise_meta = add_projection_noise(
        series=cloud_proj,
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        enable=bool(args.projection_noise_enable),
        strength=float(args.projection_noise_strength),
        ar1=float(args.projection_noise_ar1),
        seed=int(args.projection_noise_seed),
        name="cloud",
        min_frac=float(args.projection_noise_min_frac),
        max_frac=float(args.projection_noise_max_frac),
    )
    temp_proj, temp_noise_meta = add_projection_noise(
        series=temp_proj,
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        enable=bool(args.projection_noise_enable),
        strength=float(args.projection_noise_strength),
        ar1=float(args.projection_noise_ar1),
        seed=int(args.projection_noise_seed),
        name="temp",
        min_frac=float(args.projection_noise_min_frac),
        max_frac=float(args.projection_noise_max_frac),
    )
    hum_proj, hum_noise_meta = add_projection_noise(
        series=hum_proj,
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        enable=bool(args.projection_noise_enable),
        strength=float(args.projection_noise_strength),
        ar1=float(args.projection_noise_ar1),
        seed=int(args.projection_noise_seed),
        name="humidity",
        min_frac=float(args.projection_noise_min_frac),
        max_frac=float(args.projection_noise_max_frac),
    )
    precip_proj, precip_noise_meta = add_projection_noise(
        series=precip_proj,
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        fit_start=fit_start,
        enable=bool(args.projection_noise_enable),
        strength=float(args.projection_noise_strength),
        ar1=float(args.projection_noise_ar1),
        seed=int(args.projection_noise_seed),
        name="precip",
        min_frac=float(args.projection_noise_min_frac),
        max_frac=float(args.projection_noise_max_frac),
    )

    annual_idx["solar_kwh_m2_day"] = clamp_physical(solar_proj, lower=0.0)
    annual_idx["cloud_pct"] = clamp_physical(cloud_proj, lower=0.0, upper=100.0)
    annual_idx["temp_c"] = temp_proj
    annual_idx["humidity_pct"] = clamp_physical(hum_proj, lower=0.0, upper=100.0)
    annual_idx["precip_mm"] = clamp_physical(precip_proj, lower=0.0)

    # 4) Derived climate indicators
    annual_idx["effective_uv_kwh_m2_day"] = annual_idx["solar_kwh_m2_day"] * (1.0 - (annual_idx["cloud_pct"] / 100.0))

    base_mask = (annual_idx.index >= int(args.baseline_start)) & (annual_idx.index <= int(args.baseline_end))
    if base_mask.sum() < 10:
        raise SystemExit("Baseline period too short after preprocessing")

    baseline_solar = float(annual_idx.loc[base_mask, "solar_kwh_m2_day"].mean())
    baseline_cloud = float(annual_idx.loc[base_mask, "cloud_pct"].mean())
    baseline_precip = float(annual_idx.loc[base_mask, "precip_mm"].mean())
    baseline_hum = float(annual_idx.loc[base_mask, "humidity_pct"].mean())
    baseline_temp = float(annual_idx.loc[base_mask, "temp_c"].mean())
    baseline_eff_uv = float(annual_idx.loc[base_mask, "effective_uv_kwh_m2_day"].mean())

    annual_idx["solar_delta_pct"] = 100.0 * (annual_idx["solar_kwh_m2_day"] - baseline_solar) / baseline_solar
    annual_idx["cloud_delta_pct"] = 100.0 * (annual_idx["cloud_pct"] - baseline_cloud) / baseline_cloud
    annual_idx["precip_delta_pct"] = 100.0 * (annual_idx["precip_mm"] - baseline_precip) / baseline_precip
    annual_idx["humidity_delta_pct"] = 100.0 * (annual_idx["humidity_pct"] - baseline_hum) / baseline_hum
    annual_idx["temp_delta_c"] = annual_idx["temp_c"] - baseline_temp
    annual_idx["effective_uv_delta_pct"] = 100.0 * (annual_idx["effective_uv_kwh_m2_day"] - baseline_eff_uv) / baseline_eff_uv

    # 5) Per-10k skin-cancer proxy scenario
    base_per10k = float((float(args.baseline_new_cases) / float(args.baseline_population)) * 10000.0)

    # WHO Q&A anchor (10% ozone reduction -> +300k NMSC over ~2-3M baseline)
    rr_low_10pct = 1.0 + (300000.0 / 3000000.0)   # ~1.10
    rr_mid_10pct = 1.0 + (300000.0 / 2500000.0)   # ~1.12
    rr_high_10pct = 1.0 + (300000.0 / 2000000.0)  # ~1.15

    beta_low = float(math.log(rr_low_10pct) / 10.0)
    beta_mid = float(math.log(rr_mid_10pct) / 10.0)
    beta_high = float(math.log(rr_high_10pct) / 10.0)

    uv_lag_years = max(1, int(args.uv_lag_years))
    annual_idx["effective_uv_delta_pct_lagged"] = (
        annual_idx["effective_uv_delta_pct"].astype(float).rolling(window=uv_lag_years, min_periods=1).mean()
    )
    x = annual_idx["effective_uv_delta_pct_lagged"].astype(float)
    annual_idx["rr_low"] = np.exp(beta_low * x)
    annual_idx["rr_mid"] = np.exp(beta_mid * x)
    annual_idx["rr_high"] = np.exp(beta_high * x)

    annual_idx["cases_per10k_low"] = base_per10k * annual_idx["rr_low"]
    annual_idx["cases_per10k_mid"] = base_per10k * annual_idx["rr_mid"]
    annual_idx["cases_per10k_high"] = base_per10k * annual_idx["rr_high"]

    annual_idx = apply_projected_growth_guardrail(
        annual_idx=annual_idx,
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
        growth_floor_pct=float(args.projected_case_growth_floor_pct),
        growth_cap_pct=float(args.projected_case_growth_cap_pct),
        align_with_solar=bool(args.align_cases_with_solar),
        solar_parallel_weight=float(args.solar_parallel_weight),
        solar_parallel_eps_pct=float(args.solar_parallel_eps_pct),
        uv_parallel_weight=float(args.uv_parallel_weight),
        case_growth_inertia=float(args.case_growth_inertia),
        growth_guardrail_decay=float(args.growth_guardrail_decay),
    )

    # Keep RR terms synchronized with guardrailed case series.
    annual_idx["rr_low"] = annual_idx["cases_per10k_low"] / base_per10k
    annual_idx["rr_mid"] = annual_idx["cases_per10k_mid"] / base_per10k
    annual_idx["rr_high"] = annual_idx["cases_per10k_high"] / base_per10k

    annual = annual_idx.reset_index()

    # 6) Statistical proof tables
    trend_rows = [
        trend_proof(annual_idx["solar_kwh_m2_day"], "solar_kwh_m2_day", "kWh/m2/day", args.history_start, args.history_end),
        trend_proof(annual_idx["cloud_pct"], "cloud_pct", "%", args.history_start, args.history_end),
        trend_proof(annual_idx["precip_mm"], "precip_mm", "mm/year", args.history_start, args.history_end),
        trend_proof(annual_idx["humidity_pct"], "humidity_pct", "%", args.history_start, args.history_end),
        trend_proof(annual_idx["temp_c"], "temp_c", "C", args.history_start, args.history_end),
        trend_proof(annual_idx["effective_uv_kwh_m2_day"], "effective_uv_kwh_m2_day", "kWh/m2/day", args.history_start, args.history_end),
        trend_proof(annual_idx["cases_per10k_mid"], "cases_per10k_mid", "per 10k", args.history_start, args.history_end),
    ]
    trend_df = pd.DataFrame(trend_rows)

    hist = annual[annual["is_projected"] == False].copy()
    corr_cols = [
        "solar_kwh_m2_day",
        "cloud_pct",
        "precip_mm",
        "humidity_pct",
        "temp_c",
        "effective_uv_kwh_m2_day",
        "cases_per10k_mid",
    ]
    corr_df = hist[corr_cols].corr(numeric_only=True)

    # Literature consistency check (directional)
    uv_tr = trend_df[trend_df["variable"] == "effective_uv_kwh_m2_day"]
    uv_dir = str(uv_tr["direction"].iloc[0]) if len(uv_tr) else "unknown"
    c_hist = annual.loc[annual["year"] == int(args.history_end), "cases_per10k_mid"]
    c_proj = annual.loc[annual["year"] == int(args.projection_end), "cases_per10k_mid"]
    if len(c_hist) and len(c_proj) and (uv_dir == "up") and (float(c_proj.iloc[0]) >= float(c_hist.iloc[0])):
        literature_consistency = "uyumlu"
    elif len(c_hist) and len(c_proj) and (uv_dir == "down") and (float(c_proj.iloc[0]) <= float(c_hist.iloc[0])):
        literature_consistency = "uyumlu"
    else:
        literature_consistency = "kismi_uyum_or_hassas"

    # 7) Write outputs
    annual_csv = out_root / "annual_climate_and_skin_cancer_proxy.csv"
    trend_csv = out_root / "trend_proof_table.csv"
    corr_csv = out_root / "historical_correlation_matrix.csv"
    report_md = out_root / "yearly_story_tr.md"
    meta_json = out_root / "run_meta.json"

    annual.to_csv(annual_csv, index=False)
    trend_df.to_csv(trend_csv, index=False)
    corr_df.to_csv(corr_csv)

    assumptions = {
        "baseline_population": float(args.baseline_population),
        "baseline_new_cases": float(args.baseline_new_cases),
        "baseline_cases_per10k": float(base_per10k),
        "rr_beta_low_per_pct": float(beta_low),
        "rr_beta_mid_per_pct": float(beta_mid),
        "rr_beta_high_per_pct": float(beta_high),
        "uv_lag_years": int(uv_lag_years),
        "projection_fit_window_years": int(args.projection_fit_window_years),
        "projection_fit_start_effective": int(fit_start),
        "projection_damping": float(args.projection_damping),
        "projected_case_growth_cap_pct": float(args.projected_case_growth_cap_pct),
        "projected_case_growth_floor_pct": float(args.projected_case_growth_floor_pct),
        "align_cases_with_solar": bool(args.align_cases_with_solar),
        "solar_parallel_weight": float(args.solar_parallel_weight),
        "solar_parallel_eps_pct": float(args.solar_parallel_eps_pct),
        "uv_parallel_weight": float(args.uv_parallel_weight),
        "case_growth_inertia": float(args.case_growth_inertia),
        "growth_guardrail_decay": float(args.growth_guardrail_decay),
        "projection_noise_enable": bool(args.projection_noise_enable),
        "projection_noise_strength": float(args.projection_noise_strength),
        "projection_noise_ar1": float(args.projection_noise_ar1),
        "projection_noise_seed": int(args.projection_noise_seed),
        "projection_noise_min_frac": float(args.projection_noise_min_frac),
        "projection_noise_max_frac": float(args.projection_noise_max_frac),
        "baseline_period": [int(args.baseline_start), int(args.baseline_end)],
        "history_period": [int(args.history_start), int(args.history_end)],
        "projection_end": int(args.projection_end),
    }

    build_markdown_report(
        out_md=report_md,
        annual=annual,
        trend_df=trend_df,
        literature_consistency=literature_consistency,
        assumptions=assumptions,
        history_end=int(args.history_end),
        projection_end=int(args.projection_end),
    )

    plot_climate_drivers(
        df=annual,
        history_end=int(args.history_end),
        out_png=fig_dir / "yearly_climate_drivers.png",
    )
    plot_skin_cases(
        df=annual,
        history_end=int(args.history_end),
        out_png=fig_dir / "skin_cancer_per10k_trend.png",
    )
    plot_evidence_dashboard(
        trend_df=trend_df,
        corr_df=corr_df,
        out_png=fig_dir / "evidence_dashboard.png",
    )

    meta = {
        "inputs": {
            "temp_local": str(temp_path),
            "humidity_local": str(hum_path),
            "precip_local": str(precip_path),
            "station_table1": str(station_table1_path),
            "use_station_latest": bool(args.use_station_latest),
            "station_min_records_per_year": int(args.station_min_records_per_year),
            "latitude": float(args.latitude),
            "longitude": float(args.longitude),
        },
        "station_annual_rows_used": int(len(station_annual)),
        "station_annual_year_min": int(station_annual["year"].min()) if len(station_annual) else None,
        "station_annual_year_max": int(station_annual["year"].max()) if len(station_annual) else None,
        "fit_local_from_nasa": {
            "temp": temp_fit.__dict__,
            "humidity": hum_fit.__dict__,
            "precip": precip_fit.__dict__,
        },
        "projection_trends": {
            "solar": solar_meta.__dict__,
            "cloud": cloud_meta.__dict__,
            "temp": temp_meta.__dict__,
            "humidity": hum_meta.__dict__,
            "precip": precip_meta.__dict__,
        },
        "projection_noise": {
            "solar": solar_noise_meta,
            "cloud": cloud_noise_meta,
            "temp": temp_noise_meta,
            "humidity": hum_noise_meta,
            "precip": precip_noise_meta,
        },
        "assumptions": assumptions,
        "literature_consistency": literature_consistency,
        "outputs": {
            "annual_csv": str(annual_csv),
            "trend_csv": str(trend_csv),
            "corr_csv": str(corr_csv),
            "report_md": str(report_md),
            "figures": {
                "climate": str(fig_dir / "yearly_climate_drivers.png"),
                "cases": str(fig_dir / "skin_cancer_per10k_trend.png"),
                "evidence": str(fig_dir / "evidence_dashboard.png"),
            },
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {annual_csv}")
    print(f"Wrote: {trend_csv}")
    print(f"Wrote: {corr_csv}")
    print(f"Wrote: {report_md}")
    print(f"Wrote: {fig_dir / 'yearly_climate_drivers.png'}")
    print(f"Wrote: {fig_dir / 'skin_cancer_per10k_trend.png'}")
    print(f"Wrote: {fig_dir / 'evidence_dashboard.png'}")
    print(f"Wrote: {meta_json}")


if __name__ == "__main__":
    main()
