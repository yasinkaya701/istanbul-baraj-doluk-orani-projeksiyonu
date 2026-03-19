#!/usr/bin/env python3
"""Analyze dam loss components trends: evaporation demand, irrigation proxy, human consumption.

Outputs under output/istanbul_dam_loss_components.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


BASE_DIR = Path("/Users/yasinkaya/Hackhaton")
ET0_MONTHLY_CSV = BASE_DIR / "output/tarim_et0_quant/tables/tarim_et0_monthly_history.csv"
CONSUMPTION_MONTHLY_CSV = BASE_DIR / "output/istanbul_dam_quant_exog/tables/istanbul_dam_model_input_monthly.csv"
RAINFALL_XLSX = BASE_DIR / "DATA/Sayısallaştırılmış Veri/Yağış_1980-2019.xlsx"

OUT_DIR = BASE_DIR / "output/istanbul_dam_loss_components"
CHARTS_DIR = OUT_DIR / "charts"


def to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)) and not math.isnan(float(value)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def calc_trend(years: np.ndarray, values: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(years) & np.isfinite(values)
    years = years[mask]
    values = values[mask]
    if years.size < 5:
        return {"slope_per_year": np.nan, "slope_per_decade": np.nan, "r2": np.nan, "p_value": np.nan}

    if stats is not None:
        res = stats.linregress(years, values)
        r2 = res.rvalue**2 if res.rvalue is not None else np.nan
        return {
            "slope_per_year": float(res.slope),
            "slope_per_decade": float(res.slope * 10.0),
            "r2": float(r2),
            "p_value": float(res.pvalue),
        }

    # Fallback: simple polyfit
    slope, intercept = np.polyfit(years, values, 1)
    pred = slope * years + intercept
    ss_res = float(np.nansum((values - pred) ** 2))
    ss_tot = float(np.nansum((values - np.nanmean(values)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "slope_per_year": float(slope),
        "slope_per_decade": float(slope * 10.0),
        "r2": float(r2),
        "p_value": np.nan,
    }


def load_et0_monthly() -> pd.DataFrame:
    df = pd.read_csv(ET0_MONTHLY_CSV, parse_dates=["date"])
    df = df.rename(columns={"date": "ds"})
    df["year"] = df["ds"].dt.year
    return df


def load_consumption_monthly() -> pd.DataFrame:
    df = pd.read_csv(CONSUMPTION_MONTHLY_CSV, parse_dates=["ds"])
    df["year"] = df["ds"].dt.year
    df["days_in_month"] = df["ds"].dt.days_in_month
    df["consumption_m3_month"] = df["consumption_mean_monthly"] * df["days_in_month"]
    return df


def load_rainfall_monthly_from_matrix() -> pd.DataFrame:
    raw = pd.read_excel(RAINFALL_XLSX)
    cols = list(raw.columns)
    # Keep first block of year columns before duplicated header
    if "Unnamed: 45" in cols:
        cutoff = cols.index("Unnamed: 45")
        cols = cols[:cutoff]
    raw = raw[cols]

    day_col = raw.columns[0]
    base_dates = pd.to_datetime(raw[day_col], errors="coerce")
    raw = raw.drop(columns=[day_col])

    # Extract year columns that look like 4-digit years
    year_cols: list[str] = []
    for c in raw.columns:
        try:
            y = int(str(c).split(".")[0])
        except Exception:
            continue
        if 1900 <= y <= 2100:
            year_cols.append(c)

    records: list[dict[str, object]] = []
    for c in year_cols:
        year = int(str(c).split(".")[0])
        vals = raw[c].apply(to_float)
        for base_date, v in zip(base_dates, vals):
            if pd.isna(base_date) or v is None:
                continue
            if v < 0:
                continue
            try:
                date = base_date.replace(year=year)
            except ValueError:
                # Skip Feb 29 on non-leap years
                if base_date.month == 2 and base_date.day == 29:
                    continue
                continue
            records.append({"date": date, "precip_mm": float(v)})

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    monthly = df.set_index("date").resample("MS")["precip_mm"].sum().reset_index()
    monthly["year"] = monthly["date"].dt.year
    return monthly.rename(columns={"date": "ds"})


def build_trend_table(
    label: str,
    series: pd.Series,
    years: pd.Series,
    unit: str,
    period_label: str,
) -> dict[str, object]:
    stats_dict = calc_trend(years.to_numpy(dtype=float), series.to_numpy(dtype=float))
    return {
        "component": label,
        "period": period_label,
        "unit": unit,
        "slope_per_year": stats_dict["slope_per_year"],
        "slope_per_decade": stats_dict["slope_per_decade"],
        "r2": stats_dict["r2"],
        "p_value": stats_dict["p_value"],
        "n_years": int(years.nunique()),
        "start_year": int(years.min()),
        "end_year": int(years.max()),
    }


def add_period_trend(
    rows: list[dict[str, object]],
    label: str,
    df: pd.DataFrame,
    year_col: str,
    value_col: str,
    unit: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    sub = df.copy()
    if start_year is not None:
        sub = sub[sub[year_col] >= start_year]
    if end_year is not None:
        sub = sub[sub[year_col] <= end_year]
    if sub.empty:
        return
    period_label = f"{int(sub[year_col].min())}-{int(sub[year_col].max())}"
    rows.append(
        build_trend_table(
            label=label,
            series=sub[value_col],
            years=sub[year_col],
            unit=unit,
            period_label=period_label,
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    et0 = load_et0_monthly()
    cons = load_consumption_monthly()
    rain = load_rainfall_monthly_from_matrix()

    # Evaporation demand (ET0) annual totals (complete years only)
    et0_counts = et0.groupby("year", as_index=False)["et0_mm_month"].count().rename(columns={"et0_mm_month": "n_months"})
    et0_full = et0_counts[et0_counts["n_months"] >= 12]["year"]
    et0_yearly = (
        et0[et0["year"].isin(et0_full)]
        .groupby("year", as_index=False)["et0_mm_month"]
        .sum()
        .rename(columns={"et0_mm_month": "et0_mm_year"})
    )

    # Irrigation proxy (ET0 - 0.8*P), annual totals for overlapping ET0 + rain
    if not rain.empty:
        ir_df = et0.merge(rain, on="ds", how="inner")
        ir_df["peff_mm"] = 0.8 * ir_df["precip_mm"]
        ir_df["irrigation_proxy_mm"] = (ir_df["et0_mm_month"] - ir_df["peff_mm"]).clip(lower=0.0)
        ir_df["year"] = ir_df["ds"].dt.year
        ir_counts = ir_df.groupby("year", as_index=False)["irrigation_proxy_mm"].count().rename(columns={"irrigation_proxy_mm": "n_months"})
        ir_full = ir_counts[ir_counts["n_months"] >= 12]["year"]
        ir_yearly = ir_df[ir_df["year"].isin(ir_full)].groupby("year", as_index=False)["irrigation_proxy_mm"].sum()
    else:
        ir_yearly = pd.DataFrame(columns=["year", "irrigation_proxy_mm"])

    # Human consumption yearly
    cons_counts = cons.groupby("year", as_index=False)["consumption_mean_monthly"].count().rename(columns={"consumption_mean_monthly": "n_months"})
    cons_full = cons_counts[cons_counts["n_months"] >= 12]["year"]
    cons_yearly = (
        cons[cons["year"].isin(cons_full)]
        .groupby("year", as_index=False)
        .agg(
            consumption_m3_day=("consumption_mean_monthly", "mean"),
            consumption_m3_year=("consumption_m3_month", "sum"),
        )
        .copy()
    )

    # Save annual components table
    annual = et0_yearly.merge(ir_yearly, on="year", how="outer").merge(cons_yearly, on="year", how="outer")
    annual.to_csv(OUT_DIR / "component_annual.csv", index=False)

    # Trend summary
    trend_rows: list[dict[str, object]] = []
    # ET0 trends
    add_period_trend(trend_rows, "Evaporation demand (ET0)", et0_yearly, "year", "et0_mm_year", "mm/year")
    add_period_trend(trend_rows, "Evaporation demand (ET0)", et0_yearly, "year", "et0_mm_year", "mm/year", start_year=1980)
    add_period_trend(trend_rows, "Evaporation demand (ET0)", et0_yearly, "year", "et0_mm_year", "mm/year", start_year=2000)

    # Irrigation proxy trends
    if not ir_yearly.empty:
        add_period_trend(trend_rows, "Irrigation proxy (ET0-0.8P)", ir_yearly, "year", "irrigation_proxy_mm", "mm/year")
        add_period_trend(trend_rows, "Irrigation proxy (ET0-0.8P)", ir_yearly, "year", "irrigation_proxy_mm", "mm/year", start_year=2000)
        add_period_trend(trend_rows, "Irrigation proxy (ET0-0.8P)", ir_yearly, "year", "irrigation_proxy_mm", "mm/year", start_year=2011)

    # Human consumption trends
    if not cons_yearly.empty:
        add_period_trend(trend_rows, "Human consumption", cons_yearly, "year", "consumption_m3_day", "m3/day")
    pd.DataFrame(trend_rows).to_csv(OUT_DIR / "trend_summary.csv", index=False)


if __name__ == "__main__":
    main()
