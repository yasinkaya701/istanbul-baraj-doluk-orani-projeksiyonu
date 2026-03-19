#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/yasinkaya/Hackhaton")


def load_openmeteo() -> pd.DataFrame:
    path = ROOT / "output" / "newdata_feature_store" / "tables" / "kandilli_openmeteo_monthly_1940_present.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_full_month"] == True].copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def load_driver_panel() -> pd.DataFrame:
    path = ROOT / "output" / "newdata_feature_store" / "tables" / "istanbul_dam_driver_panel.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def monthly_clim_trend(df: pd.DataFrame, value_col: str, start_year: int, end_year: int) -> pd.DataFrame:
    sub = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    rows = []
    for m in range(1, 13):
        g = sub[sub["month"] == m]
        g = g.dropna(subset=[value_col])
        if len(g) < 5:
            rows.append({"month": m, "clim": np.nan, "slope": 0.0})
            continue
        x = g["year"].astype(float).values
        y = g[value_col].astype(float).values
        slope, intercept = np.polyfit(x, y, 1)
        clim = np.nanmean(y)
        rows.append({"month": m, "clim": clim, "slope": slope})
    out = pd.DataFrame(rows)
    return out


def project_future(monthly_trend: pd.DataFrame, years: list[int], base_year: float) -> pd.DataFrame:
    rows = []
    trend_map = {int(r["month"]): (r["clim"], r["slope"]) for _, r in monthly_trend.iterrows()}
    for y in years:
        for m in range(1, 13):
            clim, slope = trend_map[m]
            val = clim + slope * (y - base_year)
            rows.append({"year": y, "month": m, "value": val})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    return df


def build_simple_water_balance(driver: pd.DataFrame, proj: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Use historical to fit delta fill = a*(rain-et0)+b
    hist = driver.dropna(subset=["weighted_total_fill", "rain_mm", "et0_mm_month"]).copy()
    hist = hist.sort_values("date")
    hist["fill_pct"] = hist["weighted_total_fill"] * 100.0
    hist["delta_fill"] = hist["fill_pct"].shift(-1) - hist["fill_pct"]
    hist["wb"] = hist["rain_mm"] - hist["et0_mm_month"]
    fit = hist.dropna(subset=["delta_fill", "wb"]).copy()
    if len(fit) < 24:
        return pd.DataFrame(), {"a": np.nan, "b": np.nan, "sigma": np.nan}
    x = fit["wb"].values
    y = fit["delta_fill"].values
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    sigma = float(np.nanstd(resid))

    # simulate from last observed fill
    last_row = hist.dropna(subset=["fill_pct"]).sort_values("date").iloc[-1]
    fill = float(last_row["fill_pct"])

    proj = proj.sort_values("date").copy()
    proj["fill_pct"] = np.nan
    proj["fill_low"] = np.nan
    proj["fill_high"] = np.nan
    for idx, row in proj.iterrows():
        wb = row["rain_mm"] - row["et0_mm_month"]
        fill = fill + a * wb + b
        fill = float(np.clip(fill, 0.0, 100.0))
        proj.at[idx, "fill_pct"] = fill
        proj.at[idx, "fill_low"] = np.clip(fill - 1.0 * sigma, 0.0, 100.0)
        proj.at[idx, "fill_high"] = np.clip(fill + 1.0 * sigma, 0.0, 100.0)
    summary = {"a": float(a), "b": float(b), "sigma": float(sigma), "start_fill": float(last_row["fill_pct"])}
    return proj, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/scientific_climate_projection_2026_2040")
    args = p.parse_args()
    out_dir = ROOT / args.out
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    openmeteo = load_openmeteo()
    driver = load_driver_panel()

    # Build humidity monthly series from driver
    hum = driver.dropna(subset=["rh_mean_pct"]).copy()
    hum = hum[["date", "year", "month", "rh_mean_pct"]]

    # Use 2010-2024 as past 15y window
    past_start = 2010
    past_end = 2024

    # Trend periods
    trend_start = 1991
    trend_end = 2024
    hum_trend_start = 2000

    temp_tr = monthly_clim_trend(openmeteo, "t_mean_c", trend_start, trend_end)
    precip_tr = monthly_clim_trend(openmeteo, "precip_mm_month", trend_start, trend_end)
    et0_tr = monthly_clim_trend(openmeteo, "et0_openmeteo_mm_month", trend_start, trend_end)
    rh_tr = monthly_clim_trend(hum, "rh_mean_pct", hum_trend_start, trend_end)

    base_year = (trend_start + trend_end) / 2.0
    future_years = list(range(2026, 2041))

    temp_future = project_future(temp_tr, future_years, base_year)
    precip_future = project_future(precip_tr, future_years, base_year)
    et0_future = project_future(et0_tr, future_years, base_year)
    rh_future = project_future(rh_tr, future_years, base_year)

    # Clamp
    precip_future["value"] = precip_future["value"].clip(lower=0)
    et0_future["value"] = et0_future["value"].clip(lower=0)
    rh_future["value"] = rh_future["value"].clip(lower=0, upper=100)

    # Past actuals
    past_open = openmeteo[(openmeteo["year"] >= past_start) & (openmeteo["year"] <= past_end)].copy()
    past_hum = hum[(hum["year"] >= past_start) & (hum["year"] <= past_end)].copy()

    # Build projection frame
    future = temp_future[["date", "value"]].rename(columns={"value": "t_mean_c"})
    future = future.merge(precip_future[["date", "value"]].rename(columns={"value": "precip_mm_month"}), on="date")
    future = future.merge(et0_future[["date", "value"]].rename(columns={"value": "et0_mm_month"}), on="date")
    future = future.merge(rh_future[["date", "value"]].rename(columns={"value": "rh_mean_pct"}), on="date", how="left")
    future["scenario"] = "future_projection"

    past = past_open[["date", "t_mean_c", "precip_mm_month", "et0_openmeteo_mm_month"]].copy()
    past = past.rename(columns={"et0_openmeteo_mm_month": "et0_mm_month"})
    past = past.merge(past_hum[["date", "rh_mean_pct"]], on="date", how="left")
    past["scenario"] = "historical"

    climate = pd.concat([past, future], ignore_index=True)
    climate = climate.sort_values("date")
    climate["climate_water_balance_mm"] = climate["precip_mm_month"] - climate["et0_mm_month"]

    climate.to_csv(out_dir / "climate_projection_2010_2040_monthly.csv", index=False)

    # Simple water balance projection
    water_proj = climate[climate["date"] >= pd.Timestamp("2026-01-01")].copy()
    water_proj = water_proj.rename(columns={"precip_mm_month": "rain_mm", "et0_mm_month": "et0_mm_month"})
    water_proj, wb_summary = build_simple_water_balance(driver, water_proj)
    water_proj.to_csv(out_dir / "simple_water_balance_projection_2026_2040.csv", index=False)

    summary = {
        "past_window": f"{past_start}-01 -> {past_end}-12",
        "future_window": "2026-01 -> 2040-12",
        "trend_period": f"{trend_start}-{trend_end}",
        "humidity_trend_period": f"{hum_trend_start}-{trend_end}",
        "wb_coeff_a": wb_summary.get("a"),
        "wb_coeff_b": wb_summary.get("b"),
        "wb_sigma": wb_summary.get("sigma"),
        "wb_start_fill_pct": wb_summary.get("start_fill"),
    }
    (out_dir / "summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")

    # Plot climate series
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(climate["date"], climate["t_mean_c"], color="#C0392B", linewidth=1.3)
    ax1.set_title("Aylik Sicaklik (C)")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(climate["date"], climate["precip_mm_month"], color="#1F77B4", linewidth=1.1)
    ax2.set_title("Aylik Yagis (mm)")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(climate["date"], climate["rh_mean_pct"], color="#2CA02C", linewidth=1.1)
    ax3.set_title("Aylik Bagil Nem (%)")

    plt.tight_layout()
    plt.savefig(fig_dir / "climate_projection_temp_precip_humidity.png", dpi=160)
    plt.close()

    # Plot ET0 and water balance
    plt.figure(figsize=(12, 5))
    plt.plot(climate["date"], climate["et0_mm_month"], color="#8E44AD", label="ET0 (mm)")
    plt.plot(climate["date"], climate["climate_water_balance_mm"], color="#16A085", label="Yagis - ET0 (mm)")
    plt.title("ET0 ve Iklim Su Dengesi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "et0_and_water_balance.png", dpi=160)
    plt.close()

    # Plot simple water balance projection
    if not water_proj.empty:
        plt.figure(figsize=(12, 5))
        plt.fill_between(water_proj["date"], water_proj["fill_low"], water_proj["fill_high"], color="#AED6F1", alpha=0.5, label="1-sigma")
        plt.plot(water_proj["date"], water_proj["fill_pct"], color="#1F618D", linewidth=1.6, label="Basit Su Dengesi")
        plt.title("Basit Su Dengesi - Doluluk Projeksiyonu (2026-2040)")
        plt.ylabel("Doluluk (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "simple_water_balance_projection.png", dpi=160)
        plt.close()

    # Plot historical fit check
    hist = driver.dropna(subset=["weighted_total_fill", "rain_mm", "et0_mm_month"]).copy()
    hist = hist.sort_values("date")
    hist["fill_pct"] = hist["weighted_total_fill"] * 100.0
    hist["wb"] = hist["rain_mm"] - hist["et0_mm_month"]
    a = wb_summary.get("a")
    b = wb_summary.get("b")
    if a is not None:
        sim = [float(hist["fill_pct"].iloc[0])]
        for wb in hist["wb"].iloc[1:]:
            sim.append(np.clip(sim[-1] + a * wb + b, 0.0, 100.0))
        hist["sim_fill"] = sim

        plt.figure(figsize=(12, 5))
        plt.plot(hist["date"], hist["fill_pct"], color="#2E86C1", label="Gozlenen")
        plt.plot(hist["date"], hist["sim_fill"], color="#A93226", alpha=0.8, label="Basit Su Dengesi")
        plt.title("Basit Su Dengesi - Tarihsel Uyum")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "simple_water_balance_backtest.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
