#!/usr/bin/env python3
"""Build application outputs for:
2) Water budget
5) Crop/planting comparison
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_application_2(out_dir: Path) -> pd.DataFrame:
    et = pd.read_csv(out_dir / "et0_inputs_completed_1987.csv")
    et["date"] = pd.to_datetime(et["date"])
    et["month"] = et["date"].dt.to_period("M").astype(str)

    et_m = (
        et.groupby("month", as_index=False)
        .agg(et0_mm=("et0_completed_mm_day", "sum"), et0_mean_mm_day=("et0_completed_mm_day", "mean"))
        .sort_values("month")
    )

    wb = pd.read_csv(out_dir / "water_balance_partial_1987.csv").sort_values("month")
    out = et_m.merge(wb[["month", "precip_obs_mm", "precip_obs_days", "coverage_flag"]], on="month", how="left")

    # Conservative effective rainfall coefficient for planning scenarios.
    out["effective_precip_mm"] = 0.8 * out["precip_obs_mm"].fillna(0.0)
    out["net_irrigation_req_mm"] = np.maximum(out["et0_mm"] - out["effective_precip_mm"], 0.0)
    out["monthly_balance_mm"] = out["effective_precip_mm"] - out["et0_mm"]
    out["cumulative_balance_mm"] = out["monthly_balance_mm"].cumsum()
    out["data_quality"] = np.where(out["coverage_flag"].eq("good"), "high", "partial")

    out = out[
        [
            "month",
            "et0_mm",
            "precip_obs_mm",
            "effective_precip_mm",
            "net_irrigation_req_mm",
            "monthly_balance_mm",
            "cumulative_balance_mm",
            "precip_obs_days",
            "coverage_flag",
            "data_quality",
        ]
    ]
    out.to_csv(out_dir / "application2_water_budget_1987.csv", index=False)
    return out


def build_application_5(out_dir: Path, wb2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wb = wb2.copy()
    wb["month_num"] = pd.to_datetime(wb["month"] + "-01").dt.month

    # Scenario-level crop calendar + Kc assumptions.
    crops = [
        {"crop": "bugday_kislik", "months": [1, 2, 3, 4, 5, 6, 11, 12], "kc": 0.95},
        {"crop": "misir", "months": [5, 6, 7, 8, 9], "kc": 1.15},
        {"crop": "domates", "months": [4, 5, 6, 7, 8, 9], "kc": 1.05},
        {"crop": "aycicegi", "months": [5, 6, 7, 8, 9], "kc": 0.95},
        {"crop": "bag_uzum", "months": [4, 5, 6, 7, 8, 9, 10], "kc": 0.85},
    ]

    monthly_rows: list[dict] = []
    summary_rows: list[dict] = []
    for spec in crops:
        sub = wb[wb["month_num"].isin(spec["months"])].copy().sort_values("month")
        if sub.empty:
            continue

        sub["crop"] = spec["crop"]
        sub["kc"] = float(spec["kc"])
        sub["etc_mm"] = sub["et0_mm"] * sub["kc"]
        sub["net_irrigation_req_crop_mm"] = np.maximum(sub["etc_mm"] - sub["effective_precip_mm"], 0.0)
        monthly_rows.extend(sub.to_dict(orient="records"))

        summary_rows.append(
            {
                "crop": spec["crop"],
                "kc": float(spec["kc"]),
                "season_months": ",".join(str(m) for m in spec["months"]),
                "season_etc_mm": float(sub["etc_mm"].sum()),
                "season_effective_precip_mm": float(sub["effective_precip_mm"].sum()),
                "season_net_irrigation_req_mm": float(np.maximum(sub["etc_mm"].sum() - sub["effective_precip_mm"].sum(), 0.0)),
                "precip_obs_days_in_season": int(sub["precip_obs_days"].sum()),
                "all_months_good_coverage": bool((sub["coverage_flag"] == "good").all()),
            }
        )

    monthly = pd.DataFrame(monthly_rows)
    summary = pd.DataFrame(summary_rows).sort_values("season_net_irrigation_req_mm", ascending=False)
    monthly.to_csv(out_dir / "application5_crop_monthly_1987.csv", index=False)
    summary.to_csv(out_dir / "application5_crop_comparison_1987.csv", index=False)
    return monthly, summary


def main() -> None:
    out_dir = Path("output/spreadsheet")
    out_dir.mkdir(parents=True, exist_ok=True)

    wb2 = build_application_2(out_dir)
    monthly, summary = build_application_5(out_dir, wb2)

    print("Wrote:", out_dir / "application2_water_budget_1987.csv")
    print("Wrote:", out_dir / "application5_crop_monthly_1987.csv")
    print("Wrote:", out_dir / "application5_crop_comparison_1987.csv")
    print()
    print("Top crop by seasonal net irrigation requirement:")
    if not summary.empty:
        row = summary.iloc[0]
        print(f"- {row['crop']}: {row['season_net_irrigation_req_mm']:.2f} mm")


if __name__ == "__main__":
    main()
