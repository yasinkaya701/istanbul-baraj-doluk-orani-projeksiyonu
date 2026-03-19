#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/yasinkaya/Hackhaton")


def month_climatology(series: pd.Series, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"val": series, "month": dates.dt.month})
    clim = df.groupby("month")["val"].median()
    return clim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_v3_newdata.csv")
    p.add_argument("--openmeteo", default="output/newdata_feature_store/tables/kandilli_openmeteo_monthly_1940_present.csv")
    p.add_argument("--out", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv")
    p.add_argument("--summary", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_summary.json")
    args = p.parse_args()

    panel = pd.read_csv(ROOT / args.panel)
    panel["date"] = pd.to_datetime(panel["date"])

    openm = pd.read_csv(ROOT / args.openmeteo)
    openm["date"] = pd.to_datetime(openm["date"])
    openm = openm[openm["is_full_month"] == True].copy()

    # Build full monthly index 2000-01 -> 2026-12
    full_dates = pd.date_range("2000-01-01", "2026-12-01", freq="MS")
    full = pd.DataFrame({"date": full_dates})
    merged = full.merge(panel, on="date", how="left")

    # Merge openmeteo for available monthly fields
    merged = merged.merge(
        openm[["date", "t_mean_c", "precip_mm_month", "et0_openmeteo_mm_month"]],
        on="date",
        how="left",
        suffixes=("_panel", "_open"),
    )

    # Fill t_mean_c with openmeteo when missing
    if "t_mean_c_panel" in merged.columns:
        merged["t_mean_c"] = merged["t_mean_c_panel"].combine_first(merged["t_mean_c_open"])
    else:
        merged["t_mean_c"] = merged["t_mean_c"].combine_first(merged["t_mean_c_open"])
    # Fill rain_mm with openmeteo precip when missing
    merged["rain_mm"] = merged["rain_mm"].combine_first(merged["precip_mm_month"])
    # Fill et0 with openmeteo et0 when missing
    merged["et0_mm_month"] = merged["et0_mm_month"].combine_first(merged["et0_openmeteo_mm_month"])

    merged = merged.drop(columns=[
        "t_mean_c_panel", "t_mean_c_open", "precip_mm_month", "et0_openmeteo_mm_month"
    ], errors="ignore")

    # Fill rh_mean_pct using monthly climatology from available history
    rh_mask = merged["rh_mean_pct"].notna()
    if rh_mask.any():
        rh_clim = month_climatology(merged.loc[rh_mask, "rh_mean_pct"], merged.loc[rh_mask, "date"])
        merged["rh_mean_pct"] = merged["rh_mean_pct"].combine_first(merged["date"].dt.month.map(rh_clim))

    # Fill pressure_kpa using monthly climatology from available history
    p_mask = merged["pressure_kpa"].notna()
    if p_mask.any():
        p_clim = month_climatology(merged.loc[p_mask, "pressure_kpa"], merged.loc[p_mask, "date"])
        merged["pressure_kpa"] = merged["pressure_kpa"].combine_first(merged["date"].dt.month.map(p_clim))

    # Recompute vpd if needed and possible
    # If vpd_kpa_mean is missing but t_mean_c and rh_mean_pct exist, approximate
    if "vpd_kpa_mean" in merged.columns:
        vpd_missing = merged["vpd_kpa_mean"].isna()
        if vpd_missing.any():
            t = merged.loc[vpd_missing, "t_mean_c"]
            rh = merged.loc[vpd_missing, "rh_mean_pct"]
            # saturation vapor pressure (kPa)
            es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
            ea = es * (rh / 100.0)
            vpd = es - ea
            merged.loc[vpd_missing, "vpd_kpa_mean"] = vpd

    merged = merged.sort_values("date")

    # Clip to 2000-01 -> 2026-12
    merged = merged[(merged["date"] >= "2000-01-01") & (merged["date"] <= "2026-12-01")].copy()

    merged.to_csv(ROOT / args.out, index=False)

    def span(col):
        s = merged.dropna(subset=[col])
        if s.empty:
            return None
        return {"start": str(s["date"].min().date()), "end": str(s["date"].max().date()), "rows": int(len(s))}

    summary = {
        "panel": str(args.panel),
        "openmeteo": str(args.openmeteo),
        "out": str(args.out),
        "rain_mm": span("rain_mm"),
        "t_mean_c": span("t_mean_c"),
        "rh_mean_pct": span("rh_mean_pct"),
        "pressure_kpa": span("pressure_kpa"),
        "et0_mm_month": span("et0_mm_month"),
        "vpd_kpa_mean": span("vpd_kpa_mean"),
        "weighted_total_fill": span("weighted_total_fill"),
    }
    (ROOT / args.summary).write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
