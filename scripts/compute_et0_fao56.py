#!/usr/bin/env python3
"""Compute daily FAO-56 reference evapotranspiration (ET0) from local datasets.

Current data adapters in this script are tailored for the project files:
  - DATA/Sayısallaştırılmış Veri/1987_Sıcaklık_Saat Başı.xlsx
  - DATA/Sayısallaştırılmış Veri/Nem-1980-2014.xlsx

The implementation supports incomplete observations by using FAO-56 fallback
assumptions for missing wind and radiation:
  - u2 (2 m wind speed): constant fallback (default 2.0 m/s)
  - Rs (solar radiation): Hargreaves temperature-range estimate
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GSC_MJ_M2_MIN = 0.0820
SIGMA_MJ_K4_M2_DAY = 4.903e-9


@dataclass(frozen=True)
class EToConfig:
    year: int
    latitude_deg: float
    elevation_m: float
    wind_u2_m_s: float
    krs: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FAO-56 ET0 from local temperature and humidity data.")
    p.add_argument(
        "--temp-xlsx",
        type=Path,
        default=Path("DATA/Sayısallaştırılmış Veri/1987_Sıcaklık_Saat Başı.xlsx"),
        help="Hourly temperature workbook path.",
    )
    p.add_argument(
        "--humidity-xlsx",
        type=Path,
        default=Path("DATA/Sayısallaştırılmış Veri/Nem-1980-2014.xlsx"),
        help="Daily humidity matrix workbook path (years in columns).",
    )
    p.add_argument("--year", type=int, default=1987, help="Target year for ET0 calculation.")
    p.add_argument("--latitude", type=float, default=41.01, help="Latitude (deg).")
    p.add_argument("--elevation-m", type=float, default=39.0, help="Station elevation above sea level (m).")
    p.add_argument(
        "--u2",
        type=float,
        default=2.0,
        help="2 m wind speed (m/s). Used as constant fallback when wind observations are unavailable.",
    )
    p.add_argument(
        "--krs",
        type=float,
        default=0.19,
        help="Hargreaves radiation coefficient (0.16 interior / 0.19 coastal).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/et0/et0_fao56_daily_1987.csv"),
        help="Daily output CSV path.",
    )
    p.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("output/et0/et0_fao56_summary_1987.json"),
        help="Summary JSON path.",
    )
    return p.parse_args()


def load_temperature_daily(temp_xlsx: Path, year: int) -> pd.DataFrame:
    df = pd.read_excel(temp_xlsx, sheet_name=0, header=0)
    if "Unnamed: 0" not in df.columns:
        raise ValueError(f"Unexpected temperature format: first column not found in {temp_xlsx}")
    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df[df["date"].dt.year == year].copy()
    if df.empty:
        raise ValueError(f"No temperature rows found for year={year} in {temp_xlsx}")

    hour_cols = [c for c in df.columns if c != "date"]
    for c in hour_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({"date": df["date"]})
    out["t_mean_c"] = df[hour_cols].mean(axis=1)
    out["t_min_c"] = df[hour_cols].min(axis=1)
    out["t_max_c"] = df[hour_cols].max(axis=1)
    return out.sort_values("date").reset_index(drop=True)


def _extract_year_col_index(humidity_raw: pd.DataFrame, year: int) -> int:
    # Year headers are on row index=1 in this workbook.
    year_row = humidity_raw.iloc[1]
    for idx, value in year_row.items():
        if pd.isna(value):
            continue
        if isinstance(value, (int, float)) and int(value) == year:
            return int(idx)
    raise ValueError(f"Year column {year} not found in humidity workbook.")


def load_humidity_daily(humidity_xlsx: Path, year: int) -> pd.DataFrame:
    raw = pd.read_excel(humidity_xlsx, sheet_name=0, header=None)
    year_col = _extract_year_col_index(raw, year)

    # Data rows start after header block. We select datetime-like rows only.
    rows = raw.iloc[:, [0, year_col]].copy()
    rows.columns = ["template_date_raw", "rh_mean_pct"]
    raw_dates = rows["template_date_raw"]
    if pd.api.types.is_datetime64_any_dtype(raw_dates):
        rows["template_date"] = raw_dates
    else:
        serial = pd.to_numeric(raw_dates, errors="coerce")
        dt_serial = pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")
        missing = dt_serial.isna()
        if missing.any():
            dt_text = pd.to_datetime(raw_dates[missing], format="mixed", errors="coerce")
            dt_serial.loc[missing] = dt_text
        rows["template_date"] = dt_serial
    rows["rh_mean_pct"] = pd.to_numeric(rows["rh_mean_pct"], errors="coerce")

    rows = rows.dropna(subset=["template_date", "rh_mean_pct"]).copy()
    rows["doy"] = rows["template_date"].dt.dayofyear
    rows["date"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(rows["doy"] - 1, unit="D")
    rows = rows[["date", "rh_mean_pct"]].sort_values("date").drop_duplicates(subset=["date"])

    # Keep only target year range (guards against template-date artifacts).
    rows = rows[rows["date"].dt.year == year].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No humidity rows found for year={year} in {humidity_xlsx}")
    return rows


def saturation_vapor_pressure_kpa(temp_c: pd.Series | np.ndarray) -> pd.Series:
    temp = np.asarray(temp_c, dtype=float)
    return pd.Series(0.6108 * np.exp((17.27 * temp) / (temp + 237.3)))


def calc_ra_mj_m2_day(doy: pd.Series, latitude_deg: float) -> pd.Series:
    lat_rad = math.radians(float(latitude_deg))
    j = doy.to_numpy(dtype=float)
    dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * j)
    solar_dec = 0.409 * np.sin((2.0 * np.pi / 365.0) * j - 1.39)
    ws_arg = -np.tan(lat_rad) * np.tan(solar_dec)
    ws_arg = np.clip(ws_arg, -1.0, 1.0)
    ws = np.arccos(ws_arg)
    ra = (24.0 * 60.0 / np.pi) * GSC_MJ_M2_MIN * dr * (
        ws * np.sin(lat_rad) * np.sin(solar_dec) + np.cos(lat_rad) * np.cos(solar_dec) * np.sin(ws)
    )
    return pd.Series(ra)


def compute_et0_fao56(df: pd.DataFrame, cfg: EToConfig) -> pd.DataFrame:
    out = df.copy()
    out["doy"] = out["date"].dt.dayofyear.astype(int)

    # Atmospheric pressure [kPa] from elevation when observed pressure is unavailable.
    p_kpa = 101.3 * ((293.0 - 0.0065 * cfg.elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * p_kpa
    out["pressure_kpa"] = p_kpa
    out["gamma_kpa_c"] = gamma

    out["es_tmax_kpa"] = saturation_vapor_pressure_kpa(out["t_max_c"])
    out["es_tmin_kpa"] = saturation_vapor_pressure_kpa(out["t_min_c"])
    out["es_kpa"] = 0.5 * (out["es_tmax_kpa"] + out["es_tmin_kpa"])

    # FAO-56 Eq.19 fallback when only RHmean is available.
    out["ea_kpa"] = (out["rh_mean_pct"] / 100.0) * out["es_kpa"]

    out["delta_kpa_c"] = (
        4098.0
        * (0.6108 * np.exp((17.27 * out["t_mean_c"]) / (out["t_mean_c"] + 237.3)))
        / ((out["t_mean_c"] + 237.3) ** 2)
    )

    out["ra_mj_m2_day"] = calc_ra_mj_m2_day(out["doy"], cfg.latitude_deg)
    out["rso_mj_m2_day"] = (0.75 + 2.0e-5 * cfg.elevation_m) * out["ra_mj_m2_day"]

    # FAO-56 temperature-range estimate for Rs.
    delta_t = np.maximum(out["t_max_c"] - out["t_min_c"], 0.0)
    out["rs_mj_m2_day"] = cfg.krs * np.sqrt(delta_t) * out["ra_mj_m2_day"]
    out["rs_mj_m2_day"] = np.minimum(out["rs_mj_m2_day"], out["rso_mj_m2_day"])

    out["rns_mj_m2_day"] = 0.77 * out["rs_mj_m2_day"]

    tmax_k = out["t_max_c"] + 273.16
    tmin_k = out["t_min_c"] + 273.16
    rs_rso = np.where(out["rso_mj_m2_day"] > 0, out["rs_mj_m2_day"] / out["rso_mj_m2_day"], np.nan)
    rs_rso = np.clip(rs_rso, 0.0, 1.0)
    out["rs_rso_ratio"] = rs_rso

    out["rnl_mj_m2_day"] = (
        SIGMA_MJ_K4_M2_DAY
        * ((tmax_k**4 + tmin_k**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(out["ea_kpa"], 0.0)))
        * (1.35 * out["rs_rso_ratio"] - 0.35)
    )

    out["rn_mj_m2_day"] = out["rns_mj_m2_day"] - out["rnl_mj_m2_day"]
    out["u2_m_s"] = cfg.wind_u2_m_s

    num = (
        0.408 * out["delta_kpa_c"] * out["rn_mj_m2_day"]
        + out["gamma_kpa_c"] * (900.0 / (out["t_mean_c"] + 273.0)) * out["u2_m_s"] * (out["es_kpa"] - out["ea_kpa"])
    )
    den = out["delta_kpa_c"] + out["gamma_kpa_c"] * (1.0 + 0.34 * out["u2_m_s"])
    out["et0_mm_day"] = np.where(den > 0, num / den, np.nan)
    out["et0_mm_day"] = out["et0_mm_day"].clip(lower=0.0)

    return out


def build_summary(df: pd.DataFrame, cfg: EToConfig) -> dict:
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(et0_mm_month=("et0_mm_day", "sum"), et0_mm_day_mean=("et0_mm_day", "mean"))
    )
    summary = {
        "config": {
            "year": cfg.year,
            "latitude_deg": cfg.latitude_deg,
            "elevation_m": cfg.elevation_m,
            "u2_m_s_constant": cfg.wind_u2_m_s,
            "krs_radiation_coeff": cfg.krs,
            "method": "FAO-56 Penman-Monteith (daily)",
            "fallbacks": [
                "Solar radiation Rs estimated from temperature range (Hargreaves coefficient kRs).",
                "Wind speed u2 set as constant fallback due missing wind observations.",
                "Actual vapor pressure ea computed from RHmean and es.",
                "Atmospheric pressure estimated from elevation.",
            ],
        },
        "stats": {
            "n_days": int(df["et0_mm_day"].notna().sum()),
            "et0_mm_day_mean": float(df["et0_mm_day"].mean()),
            "et0_mm_day_min": float(df["et0_mm_day"].min()),
            "et0_mm_day_max": float(df["et0_mm_day"].max()),
            "et0_mm_year_sum": float(df["et0_mm_day"].sum()),
        },
        "monthly": monthly.to_dict(orient="records"),
    }
    return summary


def main() -> None:
    args = parse_args()
    cfg = EToConfig(
        year=int(args.year),
        latitude_deg=float(args.latitude),
        elevation_m=float(args.elevation_m),
        wind_u2_m_s=float(args.u2),
        krs=float(args.krs),
    )

    temp = load_temperature_daily(args.temp_xlsx, cfg.year)
    hum = load_humidity_daily(args.humidity_xlsx, cfg.year)
    merged = temp.merge(hum, on="date", how="inner")
    if merged.empty:
        raise SystemExit("No overlapping temperature and humidity rows after merge.")

    out = compute_et0_fao56(merged, cfg)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = build_summary(out, cfg)
    args.out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ET0 daily rows: {len(out)}")
    print(f"ET0 yearly total (mm): {summary['stats']['et0_mm_year_sum']:.2f}")
    print(f"ET0 mean daily (mm/day): {summary['stats']['et0_mm_day_mean']:.3f}")
    print(f"Wrote daily CSV: {args.out_csv}")
    print(f"Wrote summary JSON: {args.out_summary_json}")


if __name__ == "__main__":
    main()
