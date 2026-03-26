#!/usr/bin/env python3
"""Build an ET0 package using the provided daily solar radiation file as input."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_et0_fao56 import calc_ra_mj_m2_day


SIGMA_MJ_K4_M2_DAY = 4.903e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ET0 package from daily radiation input.")
    parser.add_argument(
        "--met-input",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/es_ea_newdata_daily.csv"),
        help="Daily meteorology input with Tmax/Tmin/es/ea columns.",
    )
    parser.add_argument(
        "--solar-input",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/universal_datasets/daily_solar_radiation_complete.csv"),
        help="Daily solar radiation CSV to use as Rs input.",
    )
    parser.add_argument(
        "--quant-script",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/scripts/quant_regime_projection.py"),
        help="Quant forecast script path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation"),
        help="Output directory.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Tarimsal",
        help="Context label used in titles (e.g., 'Baraj' or 'Tarimsal').",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="tarim_et0",
        help="Filename prefix for generated artifacts.",
    )
    parser.add_argument("--latitude", type=float, default=41.01, help="Latitude in degrees.")
    parser.add_argument("--elevation-m", type=float, default=39.0, help="Elevation in meters.")
    parser.add_argument("--u2", type=float, default=2.0, help="Constant 2 m wind speed in m/s.")
    parser.add_argument("--target-year", type=int, default=2035, help="Forecast horizon year.")
    return parser.parse_args()


def saturation_vapor_pressure_kpa(temp_c: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))


def delta_svp_curve_kpa_c(temp_c: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    return 4098.0 * (0.6108 * np.exp((17.27 * temp) / (temp + 237.3))) / ((temp + 237.3) ** 2)


def pressure_from_elevation_kpa(elevation_m: float) -> float:
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26


def build_daily_history(met_input: Path, solar_input: Path, latitude: float, elevation_m: float, u2_m_s: float) -> pd.DataFrame:
    met = pd.read_csv(
        met_input,
        usecols=["date", "t_max_c", "t_min_c", "es_kpa", "ea_kpa", "vpd_kpa"],
        parse_dates=["date"],
    )
    met = met.dropna(subset=["date", "t_max_c", "t_min_c", "es_kpa", "ea_kpa"]).copy()
    met["t_mean_c"] = (pd.to_numeric(met["t_max_c"], errors="coerce") + pd.to_numeric(met["t_min_c"], errors="coerce")) / 2.0
    met = met.dropna(subset=["t_mean_c"]).copy()

    solar = pd.read_csv(
        solar_input,
        usecols=["date", "daily_total_mj_m2", "data_source", "source_file"],
        parse_dates=["date"],
    )
    solar = solar.dropna(subset=["date", "daily_total_mj_m2"]).copy()
    solar = solar.rename(
        columns={
            "daily_total_mj_m2": "rs_raw_mj_m2_day",
            "data_source": "rs_data_source",
        }
    )
    solar["rs_raw_mj_m2_day"] = pd.to_numeric(solar["rs_raw_mj_m2_day"], errors="coerce")
    solar = solar.dropna(subset=["rs_raw_mj_m2_day"]).sort_values("date")
    solar = solar.drop_duplicates(subset=["date"], keep="last")

    daily = met.merge(solar, on="date", how="inner").sort_values("date").reset_index(drop=True)
    daily["doy"] = daily["date"].dt.dayofyear.astype(int)
    daily["ra_mj_m2_day"] = calc_ra_mj_m2_day(daily["doy"], latitude).to_numpy(dtype=float)
    daily["rso_mj_m2_day"] = (0.75 + 2.0e-5 * elevation_m) * daily["ra_mj_m2_day"]
    daily["rs_mj_m2_day"] = np.minimum(daily["rs_raw_mj_m2_day"], daily["rso_mj_m2_day"])
    daily["rs_clipped"] = daily["rs_raw_mj_m2_day"] > daily["rso_mj_m2_day"]

    daily["p_kpa"] = pressure_from_elevation_kpa(elevation_m)
    daily["gamma_kpa_c"] = 0.000665 * daily["p_kpa"]
    daily["delta_kpa_c"] = delta_svp_curve_kpa_c(daily["t_mean_c"])
    daily["u2_m_s"] = float(u2_m_s)
    daily["u2_source"] = "constant_fao56_fallback"
    daily["g_mj_m2_day"] = 0.0

    daily["rns_mj_m2_day"] = 0.77 * daily["rs_mj_m2_day"]
    rs_rso = np.where(daily["rso_mj_m2_day"] > 0, daily["rs_mj_m2_day"] / daily["rso_mj_m2_day"], np.nan)
    rs_rso = np.clip(rs_rso, 0.0, 1.0)
    daily["rs_rso_ratio"] = rs_rso

    tmax_k = daily["t_max_c"] + 273.16
    tmin_k = daily["t_min_c"] + 273.16
    daily["rnl_mj_m2_day"] = (
        SIGMA_MJ_K4_M2_DAY
        * ((tmax_k**4 + tmin_k**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(daily["ea_kpa"], 0.0)))
        * (1.35 * daily["rs_rso_ratio"] - 0.35)
    )
    daily["rn_mj_m2_day"] = daily["rns_mj_m2_day"] - daily["rnl_mj_m2_day"]

    numerator = (
        0.408 * daily["delta_kpa_c"] * (daily["rn_mj_m2_day"] - daily["g_mj_m2_day"])
        + daily["gamma_kpa_c"] * (900.0 / (daily["t_mean_c"] + 273.0)) * daily["u2_m_s"] * daily["vpd_kpa"]
    )
    denominator = daily["delta_kpa_c"] + daily["gamma_kpa_c"] * (1.0 + 0.34 * daily["u2_m_s"])
    daily["et0_mm_day"] = np.where(denominator > 0, numerator / denominator, np.nan)
    daily["et0_mm_day"] = daily["et0_mm_day"].clip(lower=0.0)
    daily["radiation_input_file"] = str(solar_input)

    cols = [
        "date",
        "t_mean_c",
        "t_max_c",
        "t_min_c",
        "es_kpa",
        "ea_kpa",
        "vpd_kpa",
        "rs_raw_mj_m2_day",
        "rs_mj_m2_day",
        "rs_clipped",
        "rs_data_source",
        "source_file",
        "ra_mj_m2_day",
        "rso_mj_m2_day",
        "rn_mj_m2_day",
        "g_mj_m2_day",
        "u2_m_s",
        "u2_source",
        "p_kpa",
        "gamma_kpa_c",
        "delta_kpa_c",
        "et0_mm_day",
        "radiation_input_file",
    ]
    return daily[cols].copy()


def build_monthly_history(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()

    monthly = (
        d.resample("MS")
        .agg(
            days_present=("et0_mm_day", "count"),
            et0_mm_month=("et0_mm_day", "sum"),
            et0_mm_day_mean=("et0_mm_day", "mean"),
            t_mean_c=("t_mean_c", "mean"),
            rs_mj_m2_day=("rs_mj_m2_day", "mean"),
            vpd_kpa=("vpd_kpa", "mean"),
            real_extracted_days=("rs_data_source", lambda s: int((s == "real_extracted").sum())),
            synthetic_days=("rs_data_source", lambda s: int((s == "synthetic").sum())),
        )
    )
    monthly["days_in_month"] = monthly.index.days_in_month.astype(int)
    monthly["coverage_frac"] = np.where(monthly["days_in_month"] > 0, monthly["days_present"] / monthly["days_in_month"], 0.0)
    monthly["is_reliable"] = monthly["coverage_frac"] >= 0.80
    monthly = monthly[monthly["days_present"] > 0].copy()
    monthly["date"] = monthly.index
    return monthly.reset_index(drop=True)


def build_yearly_history(monthly: pd.DataFrame) -> pd.DataFrame:
    y = monthly.copy()
    y["year"] = pd.to_datetime(y["date"]).dt.year
    out = (
        y.groupby("year", as_index=False)
        .agg(
            months_present=("et0_mm_month", "count"),
            reliable_months=("is_reliable", "sum"),
            et0_mm_year=("et0_mm_month", "sum"),
            et0_mm_day_mean=("et0_mm_day_mean", "mean"),
            t_mean_c=("t_mean_c", "mean"),
            rs_mj_m2_day=("rs_mj_m2_day", "mean"),
            vpd_kpa=("vpd_kpa", "mean"),
            real_extracted_days=("real_extracted_days", "sum"),
            synthetic_days=("synthetic_days", "sum"),
        )
    )
    out = out[out["reliable_months"] == 12].copy()
    out["date"] = pd.to_datetime(out["year"].astype(str) + "-01-01")
    cols = [
        "date",
        "year",
        "months_present",
        "reliable_months",
        "et0_mm_year",
        "et0_mm_day_mean",
        "t_mean_c",
        "rs_mj_m2_day",
        "vpd_kpa",
        "real_extracted_days",
        "synthetic_days",
    ]
    return out[cols]


def plot_history(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    charts_dir: Path,
    label: str,
    prefix: str,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(d.index, d["et0_mm_day"], color="#d97b29", linewidth=0.55, alpha=0.25, label="Günlük ET0")
    ax.plot(d.index, d["et0_mm_day"].rolling(30, min_periods=10).mean(), color="#8d3b0d", linewidth=1.8, label="30 gün ort.")
    ax.set_title(f"Günlük {label} ET0")
    ax.set_ylabel("ET0 (mm/gün)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / f"{prefix}_daily_history.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    m = monthly.copy()
    m["date"] = pd.to_datetime(m["date"])
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(m["date"], m["et0_mm_month"], color="#a44a3f", linewidth=0.9, alpha=0.45, label="Aylık ET0")
    ax.plot(m["date"], m["et0_mm_month"].rolling(12, min_periods=6).mean(), color="#5d1f1f", linewidth=2.0, label="12 ay ort.")
    ax.set_title(f"Aylık {label} ET0")
    ax.set_ylabel("ET0 (mm/ay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / f"{prefix}_monthly_history.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    y = yearly.copy()
    coef = np.polyfit(y["year"], y["et0_mm_year"], 1)
    trend = np.polyval(coef, y["year"])
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.bar(y["year"], y["et0_mm_year"], color="#5b8e7d", width=0.9, alpha=0.85)
    ax.plot(y["year"], trend, color="#1d3557", linewidth=2.0, label=f"Trend: {coef[0] * 10.0:+.1f} mm/10y")
    ax.set_title(f"Yıllık {label} ET0")
    ax.set_ylabel("ET0 (mm/yıl)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / f"{prefix}_yearly_trend.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_quant_forecast(monthly_source_csv: Path, quant_script: Path, quant_dir: Path, target_year: int) -> Path:
    quant_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = quant_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(tmp_dir))
    env.setdefault("MPLCONFIGDIR", str(quant_dir / ".mpl"))
    env.setdefault("XDG_CACHE_HOME", str(quant_dir / ".cache"))
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    cmd = [
        "python3",
        str(quant_script),
        "--observations",
        str(monthly_source_csv),
        "--output-dir",
        str(quant_dir),
        "--input-kind",
        "single",
        "--timestamp-col",
        "date",
        "--value-col",
        "et0_mm_month",
        "--single-variable",
        "et0",
        "--variables",
        "et0",
        "--target-year",
        str(target_year),
        "--disable-climate-adjustment",
        "--holdout-steps",
        "12",
        "--backtest-splits",
        "3",
        "--min-train-steps",
        "60",
        "--vol-model",
        "ewma",
    ]
    subprocess.run(cmd, check=True, env=env)

    matches = sorted((quant_dir / "forecasts").glob(f"et0*_quant_to_{target_year}.csv"))
    if not matches:
        raise FileNotFoundError(f"Forecast CSV not found in {quant_dir / 'forecasts'}")
    return matches[0]


def clean_forecast_csv(raw_forecast_csv: Path, out_csv: Path) -> pd.DataFrame:
    fc = pd.read_csv(raw_forecast_csv, parse_dates=["ds"])
    for col in ["actual", "yhat", "yhat_lower", "yhat_upper"]:
        if col in fc.columns:
            fc[col] = pd.to_numeric(fc[col], errors="coerce").clip(lower=0.0)
    fc = fc.rename(columns={"ds": "date"})
    if "unit" in fc.columns:
        fc["unit"] = "mm"
    fc.to_csv(out_csv, index=False)
    return fc


def plot_forecast(forecast_df: pd.DataFrame, charts_dir: Path, label: str, prefix: str) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    hist = df[df["is_forecast"] == False].copy()
    fc = df[df["is_forecast"] == True].copy()

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(hist["date"], hist["actual"], color="#4c6a92", linewidth=0.8, alpha=0.35, label="Tarihsel ET0")
    ax.plot(hist["date"], hist["yhat"], color="#1d3557", linewidth=1.6, label="Model uyumu")
    if not fc.empty:
        ax.plot(fc["date"], fc["yhat"], color="#d1495b", linewidth=2.0, label="Quant öngörü")
        if {"yhat_lower", "yhat_upper"}.issubset(fc.columns):
            ax.fill_between(fc["date"], fc["yhat_lower"], fc["yhat_upper"], color="#d1495b", alpha=0.18, label="Güven bandı")
    ax.set_title(f"{label} ET0 Quant Öngörü")
    ax.set_ylabel("ET0 (mm/ay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / f"{prefix}_quant_forecast.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_summary(daily: pd.DataFrame, monthly: pd.DataFrame, yearly: pd.DataFrame, forecast: pd.DataFrame, solar_input: Path) -> dict:
    y = yearly.copy()
    slope_per_year = np.polyfit(y["year"], y["et0_mm_year"], 1)[0]
    last_hist_year = int(y["year"].max())
    baseline_start = max(int(y["year"].min()), last_hist_year - 9)
    baseline = y[y["year"].between(baseline_start, last_hist_year)]["et0_mm_year"].mean()

    fc = forecast.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    fc["year"] = fc["date"].dt.year
    fc_future = fc[fc["is_forecast"] == True].copy()
    fc_yearly = fc_future.groupby("year", as_index=False)["yhat"].sum().rename(columns={"yhat": "et0_mm_year"})
    far_future = fc_yearly[fc_yearly["year"].between(2031, 2035)]["et0_mm_year"].mean()

    return {
        "coverage": {
            "history_rows_daily": int(len(daily)),
            "history_rows_monthly": int(len(monthly)),
            "history_rows_yearly": int(len(yearly)),
            "daily_start": str(pd.to_datetime(daily["date"]).min().date()),
            "daily_end": str(pd.to_datetime(daily["date"]).max().date()),
            "model_start": str(pd.to_datetime(monthly["date"]).min().date()),
            "model_end": str(pd.to_datetime(monthly["date"]).max().date()),
        },
        "radiation_input": {
            "file": str(solar_input),
            "real_extracted_days": int((daily["rs_data_source"] == "real_extracted").sum()),
            "synthetic_days": int((daily["rs_data_source"] == "synthetic").sum()),
            "days_clipped_to_rso": int(daily["rs_clipped"].sum()),
        },
        "historical_stats": {
            "et0_mm_year_mean": float(y["et0_mm_year"].mean()),
            "et0_mm_year_min": float(y["et0_mm_year"].min()),
            "et0_mm_year_max": float(y["et0_mm_year"].max()),
            "trend_mm_per_decade": float(slope_per_year * 10.0),
        },
        "forecast_stats": {
            "baseline_year_range": f"{baseline_start}-{last_hist_year}",
            "baseline_mm_year": float(baseline) if pd.notna(baseline) else None,
            "forecast_2031_2035_mm_year": float(far_future) if pd.notna(far_future) else None,
            "delta_2031_2035_vs_baseline_mm_year": (
                float(far_future - baseline) if pd.notna(far_future) and pd.notna(baseline) else None
            ),
        },
        "assumptions": {
            "tmean_method": "Tmean = (Tmax + Tmin) / 2",
            "delta_method": "Delta from Tmean with FAO-56 slope equation",
            "soil_heat_flux": "G = 0",
            "wind_speed": "u2 = 2.0 m/s constant fallback",
            "pressure": "elevation-derived constant pressure",
            "radiation": "provided daily radiation file used directly",
            "monthly_coverage_rule": "only months with coverage_frac >= 0.80 are used for the forecast model",
            "forecast_method": "quant regime forecast run directly on historical monthly ET0",
        },
    }


def write_report(
    report_path: Path,
    daily_csv: Path,
    monthly_csv: Path,
    yearly_csv: Path,
    forecast_csv: Path,
    charts_dir: Path,
    summary: dict,
    label: str,
) -> None:
    hist = summary["historical_stats"]
    fc = summary["forecast_stats"]
    rad = summary["radiation_input"]

    lines = [
        f"# {label} ET0 - Gerçek Radyasyon Girdili Paket",
        "",
        "## Kullandığımız Radyasyon Dosyası",
        "",
        f"- Dosya: `{rad['file']}`",
        f"- Günlük veri kapsamı: `{summary['coverage']['daily_start']}` -> `{summary['coverage']['daily_end']}`",
        f"- Model kapsamı: `{summary['coverage']['model_start']}` -> `{summary['coverage']['model_end']}`",
        f"- `real_extracted` gün: `{rad['real_extracted_days']}`",
        f"- `synthetic` gün: `{rad['synthetic_days']}`",
        "",
        "Bu dosya kullanıcının verdiği radyasyon girdisi olarak doğrudan ET0 hesabına sokuldu.",
        "Not: Dosya içindeki `data_source` kolonu korunmuştur; yani hangi günün gerçek çıkarım, hangisinin sentetik doldurma olduğu tabloda görülür.",
        "",
        "## Kabuller",
        "",
        "1. `Tmean = (Tmax + Tmin) / 2` kullanıldı.",
        "2. `Delta`, Tmean üzerinden FAO-56 eğri eğimiyle hesaplandı.",
        "3. `G = 0` alındı.",
        "4. `u2 = 2.0 m/s` sabit rüzgar kullanıldı.",
        "5. Basınç rakımdan sabit türetildi.",
        "6. Radyasyon olarak kullanıcının verdiği günlük seri kullanıldı.",
        "7. Aylık modelde sadece en az %80 gün kapsamasına sahip aylar kullanıldı.",
        "8. Gelecek öngörüsü ET0 serisinin kendisi üzerinden quant model ile yapıldı.",
        "",
        "## Temel Bulgular",
        "",
        f"- Ortalama yıllık ET0: `{hist['et0_mm_year_mean']:.1f} mm/yıl`",
        f"- Yıllık ET0 trendi: `{hist['trend_mm_per_decade']:+.1f} mm/10y`",
        f"- Min yıllık ET0: `{hist['et0_mm_year_min']:.1f} mm/yıl`",
        f"- Max yıllık ET0: `{hist['et0_mm_year_max']:.1f} mm/yıl`",
        f"- Baz dönem ({fc['baseline_year_range']}) ortalama yıllık ET0: `{fc['baseline_mm_year']:.1f} mm/yıl`",
        f"- 2031-2035 quant öngörü ortalama yıllık ET0: `{fc['forecast_2031_2035_mm_year']:.1f} mm/yıl`",
        f"- Beklenen fark: `{fc['delta_2031_2035_vs_baseline_mm_year']:+.1f} mm/yıl`",
        "",
        "## Üretilen Dosyalar",
        "",
        f"- Günlük ET0: `{daily_csv}`",
        f"- Aylık ET0: `{monthly_csv}`",
        f"- Yıllık ET0: `{yearly_csv}`",
        f"- Quant forecast: `{forecast_csv}`",
        f"- Grafikler: `{charts_dir}`",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    charts_dir = args.out_dir / "charts"
    reports_dir = args.out_dir / "reports"
    quant_dir = args.out_dir / "quant"
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    daily = build_daily_history(args.met_input, args.solar_input, args.latitude, args.elevation_m, args.u2)
    monthly_all = build_monthly_history(daily)
    yearly = build_yearly_history(monthly_all)
    valid_years = set(yearly["year"].astype(int).tolist())
    monthly = monthly_all[
        pd.to_datetime(monthly_all["date"]).dt.year.isin(valid_years) & monthly_all["is_reliable"]
    ].copy()

    daily_csv = tables_dir / f"{args.prefix}_daily_radiation_complete.csv"
    monthly_csv = tables_dir / f"{args.prefix}_monthly_radiation_complete.csv"
    yearly_csv = tables_dir / f"{args.prefix}_yearly_radiation_complete.csv"
    daily.to_csv(daily_csv, index=False)
    monthly.to_csv(monthly_csv, index=False)
    yearly.to_csv(yearly_csv, index=False)

    plot_history(daily, monthly, yearly, charts_dir, args.label, args.prefix)

    quant_source_csv = tables_dir / f"{args.prefix}_quant_source.csv"
    monthly[["date", "et0_mm_month"]].to_csv(quant_source_csv, index=False)
    raw_forecast_csv = run_quant_forecast(quant_source_csv, args.quant_script, quant_dir, args.target_year)
    forecast_csv = tables_dir / f"{args.prefix}_quant_forecast_to_{args.target_year}.csv"
    forecast_df = clean_forecast_csv(raw_forecast_csv, forecast_csv)
    plot_forecast(forecast_df, charts_dir, args.label, args.prefix)

    summary = build_summary(daily, monthly, yearly, forecast_df, args.solar_input)
    summary_path = reports_dir / f"{args.prefix}_real_radiation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / f"{args.prefix}_real_radiation_report.md"
    write_report(report_path, daily_csv, monthly_csv, yearly_csv, forecast_csv, charts_dir, summary, args.label)

    print(f"Wrote: {daily_csv}")
    print(f"Wrote: {monthly_csv}")
    print(f"Wrote: {yearly_csv}")
    print(f"Wrote: {forecast_csv}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
