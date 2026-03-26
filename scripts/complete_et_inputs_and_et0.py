#!/usr/bin/env python3
"""Complete ET input dataset and compute daily ET0 scenarios.

Outputs:
  - completed daily inputs + ET0 as CSV/XLSX
  - summary JSON with coverage and ET0 comparison
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from compute_et0_fao56 import (
    calc_ra_mj_m2_day,
    load_humidity_daily,
    load_temperature_daily,
    saturation_vapor_pressure_kpa,
)


SIGMA_MJ_K4_M2_DAY = 4.903e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build completed ET inputs and compare ET0 scenarios.")
    p.add_argument("--year", type=int, default=1987)
    p.add_argument("--latitude", type=float, default=41.01)
    p.add_argument("--elevation-m", type=float, default=39.0)
    p.add_argument(
        "--temp-xlsx",
        type=Path,
        default=Path("DATA/Sayısallaştırılmış Veri/1987_Sıcaklık_Saat Başı.xlsx"),
    )
    p.add_argument(
        "--humidity-xlsx",
        type=Path,
        default=Path("DATA/Sayısallaştırılmış Veri/Nem-1980-2014.xlsx"),
    )
    p.add_argument(
        "--nasa-json",
        type=Path,
        default=Path("tmp/nasa_power_1987.json"),
        help="NASA POWER daily JSON (WS2M, ALLSKY_SFC_SW_DWN, PS etc).",
    )
    p.add_argument(
        "--wind-graph-dir",
        type=Path,
        default=Path("output/universal_datasets_v2/wind_speed/1H"),
        help="Optional wind-speed graph-derived parquet directory for comparison.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/spreadsheet/et0_inputs_completed_1987.csv"),
    )
    p.add_argument(
        "--out-xlsx",
        type=Path,
        default=Path("output/spreadsheet/et0_inputs_completed_1987.xlsx"),
    )
    p.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("output/spreadsheet/et0_completion_summary_1987.json"),
    )
    return p.parse_args()


def load_nasa_power_daily(nasa_json: Path, year: int) -> pd.DataFrame:
    payload = json.loads(nasa_json.read_text(encoding="utf-8"))
    param = payload["properties"]["parameter"]

    def series(name: str) -> pd.Series:
        s = pd.Series(param.get(name, {}), dtype=float)
        if s.empty:
            return pd.Series(dtype=float)
        s.index = pd.to_datetime(s.index, format="%Y%m%d", errors="coerce")
        return s.sort_index()

    out = pd.DataFrame(
        {
            "date": pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"),
        }
    )
    out["t_mean_nasa_c"] = out["date"].map(series("T2M"))
    out["t_max_nasa_c"] = out["date"].map(series("T2M_MAX"))
    out["t_min_nasa_c"] = out["date"].map(series("T2M_MIN"))
    out["rh_nasa_pct"] = out["date"].map(series("RH2M"))
    out["u2_nasa_m_s"] = out["date"].map(series("WS2M"))
    out["rs_nasa_mj_m2_day"] = out["date"].map(series("ALLSKY_SFC_SW_DWN"))
    out["p_nasa_kpa"] = out["date"].map(series("PS"))

    # NASA POWER fill values are negative sentinels for missing data.
    for c in out.columns:
        if c == "date":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] < -900, c] = np.nan

    return out


def load_wind_graph_daily(wind_graph_dir: Path, year: int) -> pd.DataFrame:
    if not wind_graph_dir.exists():
        return pd.DataFrame(columns=["date", "u2_graph_raw"])

    rows: list[dict] = []
    for pq in sorted(wind_graph_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue
        if df.empty or "timestamp" not in df.columns or "value" not in df.columns:
            continue
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue
        start = ts.min().floor("D")
        end = start + pd.Timedelta(days=1)
        sub = df[(pd.to_datetime(df["timestamp"], errors="coerce") >= start) & (pd.to_datetime(df["timestamp"], errors="coerce") < end)]
        v = pd.to_numeric(sub["value"], errors="coerce")
        if v.notna().sum() == 0:
            continue
        rows.append({"date": start, "u2_graph_raw": float(v.mean())})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "u2_graph_raw"])
    out = out.drop_duplicates(subset=["date"]).sort_values("date")
    out = out[out["date"].dt.year == year].reset_index(drop=True)
    return out


def compute_et0(df: pd.DataFrame, lat_deg: float, elevation_m: float) -> pd.Series:
    doy = df["date"].dt.dayofyear.astype(int)
    ra = calc_ra_mj_m2_day(doy, lat_deg)
    rso = (0.75 + 2.0e-5 * elevation_m) * ra
    rs = np.minimum(df["rs_mj_m2_day"], rso)
    rns = 0.77 * rs

    es_tmax = saturation_vapor_pressure_kpa(df["t_max_c"]).to_numpy(dtype=float)
    es_tmin = saturation_vapor_pressure_kpa(df["t_min_c"]).to_numpy(dtype=float)
    es = 0.5 * (es_tmax + es_tmin)
    ea = (df["rh_mean_pct"].to_numpy(dtype=float) / 100.0) * es

    tmax_k = df["t_max_c"].to_numpy(dtype=float) + 273.16
    tmin_k = df["t_min_c"].to_numpy(dtype=float) + 273.16
    rs_rso = np.where(rso > 0, rs / rso, np.nan)
    rs_rso = np.clip(rs_rso, 0.0, 1.0)
    rnl = (
        SIGMA_MJ_K4_M2_DAY
        * ((tmax_k**4 + tmin_k**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
        * (1.35 * rs_rso - 0.35)
    )
    rn = rns - rnl

    p_kpa = df["p_kpa"].to_numpy(dtype=float)
    gamma = 0.000665 * p_kpa
    tmean = df["t_mean_c"].to_numpy(dtype=float)
    delta = (
        4098.0
        * (0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3)))
        / ((tmean + 237.3) ** 2)
    )
    u2 = df["u2_m_s"].to_numpy(dtype=float)

    num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)
    et0 = np.where(den > 0, num / den, np.nan)
    et0 = np.clip(et0, 0.0, None)
    return pd.Series(et0, index=df.index)


def main() -> None:
    args = parse_args()

    temp = load_temperature_daily(args.temp_xlsx, args.year)
    rh = load_humidity_daily(args.humidity_xlsx, args.year)
    nasa = load_nasa_power_daily(args.nasa_json, args.year)
    wgraph = load_wind_graph_daily(args.wind_graph_dir, args.year)

    base = pd.DataFrame({"date": pd.date_range(f"{args.year}-01-01", f"{args.year}-12-31", freq="D")})
    df = (
        base.merge(temp, on="date", how="left")
        .merge(rh, on="date", how="left")
        .merge(nasa, on="date", how="left")
        .merge(wgraph, on="date", how="left")
    )

    # Completed scenario: prioritize local T/RH, NASA for wind/radiation/pressure.
    df["t_mean_c"] = df["t_mean_c"].combine_first(df["t_mean_nasa_c"])
    df["t_min_c"] = df["t_min_c"].combine_first(df["t_min_nasa_c"])
    df["t_max_c"] = df["t_max_c"].combine_first(df["t_max_nasa_c"])
    df["rh_mean_pct"] = df["rh_mean_pct"].combine_first(df["rh_nasa_pct"])
    df["u2_m_s"] = df["u2_nasa_m_s"]
    df["rs_mj_m2_day"] = df["rs_nasa_mj_m2_day"]
    # prefer observed pressure if available, fallback to elevation-derived pressure
    p_elev_kpa = 101.3 * ((293.0 - 0.0065 * args.elevation_m) / 293.0) ** 5.26
    df["p_kpa"] = df["p_nasa_kpa"].fillna(p_elev_kpa)

    # Baseline fallback scenario for comparison.
    delta_t = np.maximum(df["t_max_c"] - df["t_min_c"], 0.0)
    ra = calc_ra_mj_m2_day(df["date"].dt.dayofyear.astype(int), args.latitude)
    rs_est = 0.19 * np.sqrt(delta_t) * ra
    rso = (0.75 + 2.0e-5 * args.elevation_m) * ra
    rs_est = np.minimum(rs_est, rso)

    baseline = pd.DataFrame(
        {
            "date": df["date"],
            "t_mean_c": df["t_mean_c"],
            "t_min_c": df["t_min_c"],
            "t_max_c": df["t_max_c"],
            "rh_mean_pct": df["rh_mean_pct"],
            "u2_m_s": 2.0,
            "rs_mj_m2_day": rs_est,
            "p_kpa": p_elev_kpa,
        }
    )

    ready = pd.DataFrame(
        {
            "date": df["date"],
            "t_mean_c": df["t_mean_c"],
            "t_min_c": df["t_min_c"],
            "t_max_c": df["t_max_c"],
            "rh_mean_pct": df["rh_mean_pct"],
            "u2_m_s": df["u2_m_s"],
            "rs_mj_m2_day": df["rs_mj_m2_day"],
            "p_kpa": df["p_kpa"],
        }
    )

    df["et0_completed_mm_day"] = compute_et0(ready, args.latitude, args.elevation_m)
    df["et0_fallback_mm_day"] = compute_et0(baseline, args.latitude, args.elevation_m)
    df["et0_diff_completed_minus_fallback"] = df["et0_completed_mm_day"] - df["et0_fallback_mm_day"]

    df["source_temp"] = np.where(df["t_mean_nasa_c"].notna() & temp.set_index("date").reindex(df["date"])["t_mean_c"].isna().to_numpy(), "nasa", "local")
    df["source_humidity"] = np.where(df["rh_nasa_pct"].notna() & rh.set_index("date").reindex(df["date"])["rh_mean_pct"].isna().to_numpy(), "nasa", "local")
    df["source_wind"] = np.where(df["u2_nasa_m_s"].notna(), "nasa", "fallback")
    df["source_radiation"] = np.where(df["rs_nasa_mj_m2_day"].notna(), "nasa", "estimated")

    out_cols = [
        "date",
        "t_mean_c",
        "t_min_c",
        "t_max_c",
        "rh_mean_pct",
        "u2_m_s",
        "rs_mj_m2_day",
        "p_kpa",
        "u2_graph_raw",
        "et0_completed_mm_day",
        "et0_fallback_mm_day",
        "et0_diff_completed_minus_fallback",
        "source_temp",
        "source_humidity",
        "source_wind",
        "source_radiation",
        "t_mean_nasa_c",
        "rh_nasa_pct",
        "u2_nasa_m_s",
        "rs_nasa_mj_m2_day",
    ]
    out_df = df[out_cols].copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    out_df.to_excel(args.out_xlsx, index=False)

    summary = {
        "year": args.year,
        "coverage": {
            "days_total": int(len(out_df)),
            "local_temp_days": int(temp["t_mean_c"].notna().sum()),
            "local_humidity_days": int(rh["rh_mean_pct"].notna().sum()),
            "nasa_wind_days": int(out_df["u2_nasa_m_s"].notna().sum()),
            "nasa_radiation_days": int(out_df["rs_nasa_mj_m2_day"].notna().sum()),
            "graph_wind_days": int(out_df["u2_graph_raw"].notna().sum()),
        },
        "et0": {
            "completed_year_sum_mm": float(out_df["et0_completed_mm_day"].sum()),
            "completed_daily_mean_mm": float(out_df["et0_completed_mm_day"].mean()),
            "fallback_year_sum_mm": float(out_df["et0_fallback_mm_day"].sum()),
            "fallback_daily_mean_mm": float(out_df["et0_fallback_mm_day"].mean()),
            "mean_diff_mm_day": float(out_df["et0_diff_completed_minus_fallback"].mean()),
        },
        "notes": [
            "Completed ET0 uses local temperature/humidity where available and NASA POWER for wind/radiation/pressure.",
            "Graph-derived wind is kept as raw comparator only; axis calibration uncertainty remains.",
            "Fallback ET0 matches the previous FAO-56 missing-data approach (u2=2 m/s, Rs from kRs).",
        ],
    }
    args.out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_xlsx}")
    print(f"Wrote: {args.out_summary_json}")
    print(
        "ET0 completed/fallback yearly totals (mm): "
        f"{summary['et0']['completed_year_sum_mm']:.2f} / {summary['et0']['fallback_year_sum_mm']:.2f}"
    )


if __name__ == "__main__":
    main()
