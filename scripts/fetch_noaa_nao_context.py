#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
OUT_DIR = Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store")
DRIVER_CSV = OUT_DIR / "tables" / "istanbul_dam_driver_panel.csv"
CLIMATE_CSV = OUT_DIR / "tables" / "istanbul_newdata_monthly_climate_panel.csv"


def fetch_nao_monthly() -> pd.DataFrame:
    text = requests.get(NAO_URL, timeout=30).text
    rows: list[dict[str, float | pd.Timestamp]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        year, month, value = int(parts[0]), int(parts[1]), float(parts[2])
        rows.append({"date": pd.Timestamp(year=year, month=month, day=1), "nao_index": value})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_driver_overlap(nao: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    driver = pd.read_csv(DRIVER_CSV, parse_dates=["date"])
    overlap = driver.merge(nao, on="date", how="inner")
    overlap["wet_season_flag"] = overlap["date"].dt.month.isin([11, 12, 1, 2, 3])
    overlap["winter_flag"] = overlap["date"].dt.month.isin([12, 1, 2])
    overlap["nao_lag1"] = overlap["nao_index"].shift(1)
    overlap["nao_lag2"] = overlap["nao_index"].shift(2)
    overlap["fill_lag1"] = overlap["weighted_total_fill"].shift(1)
    overlap["fill_change_pp"] = 100.0 * (overlap["weighted_total_fill"] - overlap["fill_lag1"])

    wet = overlap[overlap["wet_season_flag"] & overlap["rain_model_mm"].notna()].copy()
    winter = overlap[overlap["winter_flag"] & overlap["rain_model_mm"].notna()].copy()

    return overlap, {
        "driver_overlap_start": str(overlap["date"].min().date()),
        "driver_overlap_end": str(overlap["date"].max().date()),
        "driver_overlap_rows": int(len(overlap)),
        "monthly_rain_corr_all": float(overlap["nao_index"].corr(overlap["rain_model_mm"])),
        "monthly_rain_corr_wet_season": float(wet["nao_index"].corr(wet["rain_model_mm"])),
        "monthly_rain_corr_winter": float(winter["nao_index"].corr(winter["rain_model_mm"])),
        "monthly_fill_corr_wet_season": float(wet["nao_index"].corr(wet["weighted_total_fill"])),
        "monthly_fill_change_corr_wet_season": float(wet["nao_index"].corr(wet["fill_change_pp"])),
    }


def build_long_climate_overlap(nao: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    climate = pd.read_csv(CLIMATE_CSV, parse_dates=["date"])
    overlap = climate.merge(nao, on="date", how="inner")
    overlap = overlap[overlap["rain_mm"].notna()].copy()
    overlap["wet_season_flag"] = overlap["date"].dt.month.isin([11, 12, 1, 2, 3])
    overlap["winter_flag"] = overlap["date"].dt.month.isin([12, 1, 2])

    seasonal = overlap[overlap["date"].dt.month.isin([12, 1, 2])].copy()
    seasonal["season_year"] = seasonal["date"].dt.year
    seasonal.loc[seasonal["date"].dt.month == 12, "season_year"] += 1
    seasonal = (
        seasonal.groupby("season_year", as_index=False)
        .agg(nao_djf_mean=("nao_index", "mean"), rain_djf_sum_mm=("rain_mm", "sum"))
        .sort_values("season_year")
        .reset_index(drop=True)
    )

    wet = overlap[overlap["wet_season_flag"]]
    winter = overlap[overlap["winter_flag"]]

    return overlap, seasonal, {
        "climate_overlap_start": str(overlap["date"].min().date()),
        "climate_overlap_end": str(overlap["date"].max().date()),
        "climate_overlap_rows": int(len(overlap)),
        "monthly_rain_corr_wet_season_long": float(wet["nao_index"].corr(wet["rain_mm"])),
        "monthly_rain_corr_winter_long": float(winter["nao_index"].corr(winter["rain_mm"])),
        "seasonal_djf_rain_corr": float(seasonal["nao_djf_mean"].corr(seasonal["rain_djf_sum_mm"])),
        "seasonal_djf_rows": int(len(seasonal)),
    }


def main() -> None:
    out_tables = OUT_DIR / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    nao = fetch_nao_monthly()
    driver_overlap, driver_summary = build_driver_overlap(nao)
    climate_overlap, seasonal, climate_summary = build_long_climate_overlap(nao)

    nao_csv = out_tables / "noaa_cpc_nao_monthly_1950_present.csv"
    driver_csv = out_tables / "noaa_nao_vs_istanbul_driver_monthly.csv"
    climate_csv = out_tables / "noaa_nao_vs_istanbul_climate_monthly.csv"
    seasonal_csv = out_tables / "noaa_nao_vs_istanbul_djf_seasonal.csv"
    nao.to_csv(nao_csv, index=False)
    driver_overlap.to_csv(driver_csv, index=False)
    climate_overlap.to_csv(climate_csv, index=False)
    seasonal.to_csv(seasonal_csv, index=False)

    summary = {
        "source": "NOAA CPC monthly NAO index",
        "source_url": NAO_URL,
        "nao_start": str(nao["date"].min().date()),
        "nao_end": str(nao["date"].max().date()),
        "nao_rows": int(len(nao)),
        "driver_panel_summary": driver_summary,
        "long_climate_summary": climate_summary,
        "notes": [
            "NAO is a regime-scale circulation feature, not a local hydrological measurement.",
            "Useful as an exogenous winter precipitation risk feature for Istanbul.",
            "Local empirical correlations here are descriptive and should be treated as feature-screening evidence, not causal proof.",
        ],
    }
    summary_path = OUT_DIR / "noaa_nao_context_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(nao_csv)
    print(driver_csv)
    print(climate_csv)
    print(seasonal_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
