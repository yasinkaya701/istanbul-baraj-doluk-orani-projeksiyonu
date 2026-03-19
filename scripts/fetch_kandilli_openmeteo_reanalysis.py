#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 41.0615
LONGITUDE = 29.0592
TIMEZONE = "Europe/Istanbul"
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "wind_speed_10m_max",
    "sunshine_duration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Kandilli-area Open-Meteo historical reanalysis daily variables and compare ET0 against the local FAO-56 series.")
    parser.add_argument("--start-date", default="1940-01-01")
    parser.add_argument("--end-date", default=(date.today() - timedelta(days=1)).isoformat())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store"),
    )
    parser.add_argument(
        "--local-et0-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_quant/tables/tarim_et0_monthly_history.csv"),
    )
    return parser.parse_args()


def fetch_daily(start_date: str, end_date: str) -> tuple[pd.DataFrame, dict]:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
        "models": "era5_seamless",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    daily = pd.DataFrame(payload["daily"])
    daily["date"] = pd.to_datetime(daily["time"])
    daily = daily.drop(columns=["time"]).rename(
        columns={
            "temperature_2m_mean": "t_mean_c",
            "temperature_2m_max": "t_max_c",
            "temperature_2m_min": "t_min_c",
            "precipitation_sum": "precip_mm_day",
            "shortwave_radiation_sum": "rs_mj_m2_day",
            "et0_fao_evapotranspiration": "et0_openmeteo_mm_day",
            "wind_speed_10m_max": "wind_speed_10m_max_m_s",
            "sunshine_duration": "sunshine_duration_s",
        }
    )
    daily["sunshine_duration_h"] = daily["sunshine_duration_s"] / 3600.0
    return daily, payload


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        daily.assign(month=lambda df: df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(
            days_covered=("date", "count"),
            t_mean_c=("t_mean_c", "mean"),
            t_max_c=("t_max_c", "mean"),
            t_min_c=("t_min_c", "mean"),
            precip_mm_month=("precip_mm_day", "sum"),
            rs_mj_m2_month=("rs_mj_m2_day", "sum"),
            et0_openmeteo_mm_month=("et0_openmeteo_mm_day", "sum"),
            wind_speed_10m_max_m_s=("wind_speed_10m_max_m_s", "mean"),
            sunshine_duration_h_month=("sunshine_duration_h", "sum"),
        )
        .rename(columns={"month": "date"})
    )
    monthly["days_in_month"] = pd.to_datetime(monthly["date"]).dt.days_in_month
    monthly["is_full_month"] = monthly["days_covered"] == monthly["days_in_month"]
    return monthly


def compare_to_local_et0(openmeteo_monthly: pd.DataFrame, local_et0_csv: Path) -> tuple[pd.DataFrame, dict]:
    local = pd.read_csv(local_et0_csv, parse_dates=["date"])
    local["date"] = pd.to_datetime(local["date"]).dt.to_period("M").dt.to_timestamp()
    local = local[["date", "et0_mm_month", "rs_mj_m2_day", "u2_m_s", "rs_source", "u2_source"]].copy()
    merged = openmeteo_monthly.merge(local, on="date", how="inner")
    merged = merged[merged["is_full_month"]].copy()
    merged["et0_month_delta_mm"] = merged["et0_openmeteo_mm_month"] - merged["et0_mm_month"]
    merged["rs_month_from_local_mj_m2"] = merged["rs_mj_m2_day"] * pd.to_datetime(merged["date"]).dt.days_in_month
    merged["rs_month_delta_mj_m2"] = merged["rs_mj_m2_month"] - merged["rs_month_from_local_mj_m2"]

    et0_corr = float(merged["et0_openmeteo_mm_month"].corr(merged["et0_mm_month"]))
    rs_corr = float(merged["rs_mj_m2_month"].corr(merged["rs_month_from_local_mj_m2"]))
    et0_mae = float((merged["et0_month_delta_mm"]).abs().mean())
    et0_bias = float(merged["et0_month_delta_mm"].mean())
    return merged, {
        "overlap_start": str(merged["date"].min().date()),
        "overlap_end": str(merged["date"].max().date()),
        "overlap_rows": int(len(merged)),
        "et0_corr": et0_corr,
        "et0_mae_mm_month": et0_mae,
        "et0_bias_mm_month": et0_bias,
        "radiation_month_corr": rs_corr,
    }


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    daily, payload = fetch_daily(args.start_date, args.end_date)
    monthly = build_monthly(daily)
    compare_df, compare_summary = compare_to_local_et0(monthly, args.local_et0_csv)

    daily_csv = out_tables / "kandilli_openmeteo_daily_1940_present.csv"
    monthly_csv = out_tables / "kandilli_openmeteo_monthly_1940_present.csv"
    compare_csv = out_tables / "kandilli_openmeteo_vs_local_et0_monthly.csv"
    daily.to_csv(daily_csv, index=False)
    monthly.to_csv(monthly_csv, index=False)
    compare_df.to_csv(compare_csv, index=False)

    summary = {
        "source": "Open-Meteo Historical Weather API",
        "model": "era5_seamless",
        "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "timezone": TIMEZONE,
        "request_window": {"start": args.start_date, "end": args.end_date},
        "daily_rows": int(len(daily)),
        "monthly_rows": int(len(monthly)),
        "openmeteo_generationtime_ms": payload.get("generationtime_ms"),
        "api_elevation_m": payload.get("elevation"),
        "api_grid_latitude": payload.get("latitude"),
        "api_grid_longitude": payload.get("longitude"),
        "compare_to_local_et0": compare_summary,
        "notes": [
            "This is a reanalysis proxy layer, not an in-situ actinograph replacement.",
            "Useful for ET0 radiation and wind sanity-checking until direct radiation observations are integrated.",
            "Monthly overlap is compared against the existing local FAO-56 ET0 history already used in the project.",
        ],
    }
    summary_path = args.out_dir / "kandilli_openmeteo_reanalysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(daily_csv)
    print(monthly_csv)
    print(compare_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
