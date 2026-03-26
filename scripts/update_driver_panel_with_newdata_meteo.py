#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/yasinkaya/Hackhaton")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel.csv")
    p.add_argument("--newdata", default="output/newdata_feature_store/tables/newdata_meteo_monthly_from_xlsx.csv")
    p.add_argument("--out", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel_v2_newdata.csv")
    args = p.parse_args()

    panel = pd.read_csv(ROOT / args.panel)
    panel["date"] = pd.to_datetime(panel["date"])

    newdata = pd.read_csv(ROOT / args.newdata)
    if not newdata.empty:
        newdata["date"] = pd.to_datetime(newdata["date"])

    merged = panel.merge(newdata, on="date", how="left", suffixes=("", "_new"))

    # overwrite with new data when available
    if "t_mean_c" in merged.columns and "t_mean_c_new" in merged.columns:
        merged["t_mean_c"] = merged["t_mean_c_new"].combine_first(merged["t_mean_c"])
    if "rh_mean_pct" in merged.columns and "rh_mean_pct_new" in merged.columns:
        merged["rh_mean_pct"] = merged["rh_mean_pct_new"].combine_first(merged["rh_mean_pct"])
    if "pressure_kpa_newdata" in merged.columns:
        merged["pressure_kpa"] = merged["pressure_kpa_newdata"].combine_first(merged["pressure_kpa"])
    if "rain_mm_newdata" in merged.columns:
        merged["rain_mm"] = merged["rain_mm_newdata"].combine_first(merged["rain_mm"])

    drop_cols = [c for c in merged.columns if c.endswith("_new") or c.endswith("_newdata")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    merged.to_csv(ROOT / args.out, index=False)


if __name__ == "__main__":
    main()
